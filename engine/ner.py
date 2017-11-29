import numpy as np
import os
import tensorflow as tf

from utilities.dataUtils import minibatches, pad_sequences, get_chunks
from utilities.commonUtils import ProgressBar
from .baseClass import BaseModel


class NERModel(BaseModel):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    # This function will create the input for tensorflow
    def createPlaceHolder(self):
        # [batch size, max length of sentence in batch]
        self.wordIdx = tf.placeholder(tf.int32, shape=[None, None],
                        name="wordIdx")

        # [batch size]
        self.seqLen = tf.placeholder(tf.int32, shape=[None],
                        name="seqLen")

        # [batch size, max length of sentence, max length of words]
        self.charIdx = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="charIdx")

        # [batch size, max length of sentence]
        self.wordLen = tf.placeholder(tf.int32, shape=[None, None],
                        name="wordLen")

        # [batch size, max length of sentence in batch]
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # dropout and learning_rate hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

    # This function will generate feed dictionary
    def generateFeed(self, words, labels=None, lr=None, dropout=None):
        # Words: list of sentences. Sentence: list of indexes of list of words.
        # Word is a list of indexes of chars
        charIdx, wordIdx = zip(*words)
        wordIdx, seqLen = pad_sequences(wordIdx, 0)
        charIdx, wordLen = pad_sequences(charIdx, pad_token=0, nlevels=2)

        # build feed dictionary
        feed = {
            self.wordIdx: wordIdx,
            self.seqLen: seqLen
        }

        feed[self.charIdx] = charIdx
        feed[self.wordLen] = wordLen

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, seqLen

    # Create embedding lookup layers
    def createWordEmbOps(self):
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _wordEmbeddings = tf.get_variable(
                        name="_wordEmbeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _wordEmbeddings = tf.Variable(
                        self.config.embeddings,
                        name="_wordEmbeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            wordEmbeddings = tf.nn.embedding_lookup(_wordEmbeddings,
                    self.wordIdx, name="wordEmbeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _charEmbeddings = tf.get_variable(
                        name="_charEmbeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                charEmbeddings = tf.nn.embedding_lookup(_charEmbeddings,
                        self.charIdx, name="_charEmbeddings")

                # put the time dimension on axis=1
                s = tf.shape(charEmbeddings)
                charEmbeddings = tf.reshape(charEmbeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                wordLen = tf.reshape(self.wordLen, shape=[s[0]*s[1]])

                # Define bidirectional lstm of chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, charEmbeddings,
                        sequence_length=wordLen, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                wordEmbeddings = tf.concat([wordEmbeddings, output], axis=-1)

        self.wordEmbeddings =  tf.nn.dropout(wordEmbeddings, self.dropout)

    def createLossOps(self):
    	# Define logits operator
    	# Hints: each word corresponds to an output of tagged name
        
    	# Bidirectional LSTM of words
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.wordEmbeddings,
                    sequence_length=self.seqLen, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        # Define fully connected layer
        with tf.variable_scope("fc_layer"):
        	# Weights, shape is [2*hiddenSize,number_of_tagged]
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            # Biases, shape is [number_of_tagged]
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            # Since this is a bidirectional lstm, output of lstm layer is 2*hidden size
            # Reshape the output to 2*hidden size
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

        # Predict label
        self.labelsPrediction = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        # Define loss, our core metrics
        logLikelihood, transParams = tf.contrib.crf.crf_log_likelihood(
                                self.logits, self.labels, self.seqLen)
        self.transParams = transParams
        self.loss = tf.reduce_mean(-logLikelihood)

        # Add loss to tensorboard
        tf.summary.scalar("loss", self.loss)

    def buildGraph(self):
        
        # Build tensorflow graph + define loss
        self.createPlaceHolder()
        self.createWordEmbOps()
        self.createLossOps()

        # Add training operator
        self.createTrainOps(self.config.lr_method, self.lr, self.loss,
                self.config.clip)

        # Initialize session
        self.initializeSession()

    def predictBatch(self, words):
    	# Words: list of sentences
        fd, sequence_lengths = self.generateFeed(words, dropout=1.0)
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
                [self.logits, self.transParams], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def executeEpoch(self, train, dev, epoch):
        # ProgressBar - copied from keras
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = ProgressBar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.generateFeed(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.runEvaluation(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics["f1"]

    def runEvaluation(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predictBatch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}

    def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predictBatch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
