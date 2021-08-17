import numpy as np
from nodevectors import Node2Vec
import gensim
import csrgraph as cg
import time
import pandas as pd


class Node2VecResizable(Node2Vec):
    def _deepwalk(self, G, node_names):
        # Adjacency matrix
        walks_t = time.time()
        if self.verbose:
            print("Making walks...", end=" ")

        self.walks = G.random_walks(
            walklen=self.walklen,
            epochs=self.epochs,
            return_weight=self.return_weight,
            neighbor_weight=self.neighbor_weight,
        )

        if self.verbose:
            print(f"Done, T={time.time() - walks_t:.2f}")
            print("Mapping Walk Names...", end=" ")

        map_t = time.time()
        self.walks = pd.DataFrame(self.walks)

        # Map nodeId -> node name
        node_dict = dict(zip(np.arange(len(node_names)), node_names))
        for col in self.walks.columns:
            self.walks[col] = self.walks[col].map(node_dict).astype(str)

        # Somehow gensim only trains on this list iterator
        # it silently mistrains on array input
        self.walks = [list(x) for x in self.walks.itertuples(False, None)]

        if self.verbose:
            print(f"Done, T={time.time() - map_t:.2f}")

        return self.walks

    def fit_partial(self, G, pretrained_model=None):
        """
        fit new nodes / sentences to pretrained node2vec model
        reference: https://github.com/VHRanger/nodevectors

        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        G : graph data
            Graph to embed
            Can be any graph type that's supported by csrgraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)

        pretrained_model: pretrained node2vec using Node2Vec.fit()
        """
        if not isinstance(G, cg.csrgraph):
            G = cg.csrgraph(G, threads=self.threads)

        if G.threads != self.threads:
            G.set_threads(self.threads)

        # Because networkx graphs are actually iterables of their nodes
        #   we do list(G) to avoid networkx 1.X vs 2.X errors
        node_names = G.names
        if type(node_names[0]) not in [
            int,
            str,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]:
            raise ValueError("Graph node names must be int or str!")

        self.walks = self._deepwalk(G, node_names)  # training data for word2vec

        if self.verbose:
            print("Training W2V...", end=" ")
            if gensim.models.word2vec.FAST_VERSION < 1:
                print(
                    "WARNING: gensim word2vec version is unoptimized"
                    "Try version 3.6 if on windows, versions 3.7 "
                    "and 3.8 have had issues"
                )
        w2v_t = time.time()

        # Train gensim word2vec model on random walks
        if pretrained_model:
            if self.verbose:
                print("Retraining W2V...", end=" ")

            self.model = pretrained_model
            nodes_before = len(self.model.wv.index2word)

            self.model.build_vocab(self.walks, update=True)
            self.model.train(self.walks)

            nodes_after = len(self.model.wv.index2word)
            if self.verbose:
                print(
                    "Num of nodes: %d (before), %d (after)"
                    % (nodes_before, nodes_after)
                )
        else:
            if self.verbose:
                print("Training New W2V...", end=" ")

            self.model = gensim.models.Word2Vec(
                sentences=self.walks, size=self.n_components, **self.w2vparams
            )

        if not self.keep_walks:
            del self.walks
        if self.verbose:
            print(f"Done, T={time.time() - w2v_t:.2f}")

    def save_w2v_model(self, out_file):
        """
        save word2vec original model (not KeyedVectors format)
        """
        self.model.save(out_file)
