#import "neurips.typ": * 
// #import "sckathach.typ": * 
// #show regex("[Gg]raph(es|s)?"): x => celeste(x.text, freq: 0.9) 
#let ub(X) = $upright(bold(#X))$
#set list(marker: [-])

#let affls = (
  tsp: (
    department: "Student, Réseaux et Services de Télécom",
    institution: "Télécom SudParis",
    location: "Évry",
    country: "France"),
    sup_tsp: (
    department: "Supervisor, Réseaux et Services de Télécom",
    institution: "Télécom SudParis",
    location: "Évry",
    country: "France"),
)

#let authors = (
  (name: "Romain MOREAU",
   affl: "tsp",
   equal: true),
  (name: "Thomas WINNINGER", 
   affl: "tsp", 
   equal: true),
   (name: "Gregory BLANC", 
   affl: "sup_tsp", 
   equal: true),
)

#show: neurips.with(
  title: [Graph Neural Network based Intrusion Detection and its Robustness against Adversarial Attacks],
  authors: (authors, affls),
  keywords: ("Machine Learning", "NeurIPS"),
  abstract: [
    The increasing number of cyberthreats necessitates advanced Intrusion Detection Systems (IDS) for effective network security, as security analysts can not handle all the threats by themselves. Traditional IDS, classified into Network IDS (NIDS) and Host-based IDS (HIDS), have limitations that can be potentially addressed through Machine Learning (ML) advancements. Many studies have tried classical ML techniques, but they have limitations capturing long and complex attacks. In addition, these techniques are very vulnerable to adversarial attack. However, a new type of ML has proven to be particularly efficient in the task of intrusion detection: Graph Neural Networks (GNNs). GNNs offer robust data analysis capabilities by leveraging the structural properties of the network, which traditional method struggle with. This work explores the application of GNNs in IDS, and  their robustness against adversarial attacks. Key GNN architectures, including Graph Convolutional Networks (GCN), Graph Sample and Aggregate (GraphSAGE), and Graph Attention Networks (GAT), are discussed in the context of their ability to capture the semantic of attacks. Furthermore, we examine the vulnerabilities of GNNs to adversarial perturbations and various attack strategies, emphasizing the importance of developing resilient defense mechanisms. We conclude with a discussion on the practical implementation of GNN-based IDS, highlighting the balance between detection accuracy and computational feasibility.
  ],
  bibliography: bibliography("main.bib"),
)

// Advanced styling sheesh, put after all #show 
#show figure: X => [
  #set align(center)
  #v(1em)
  #X 
  #v(1em)
]

= Introduction
To fight against cyberthreat and protect network systems, there are many Intrusion Detection System -- IDS -- to monitor single or multiple hosts. The most general classifications are Network IDS (NIDS) and Host-based IDS (HIDS). The first one performs analysis on traffic to and from all devices of the network, it safeguards every device from unauthorized access. An example is to put a NIDS along the firewall to watch if anyone tries to break it. For HIDS, this one will monitor packets coming to and from a device, and it will alert the administrator if something is suspicious. 

However, with recent technologies and the progresses in Machine Learning (ML), it might be possible to improve intrusion detection with neural networks. Due to hidden layers and non-linear modelling, neural network-based IDS allows analysing huge volumes of data. But if machine learning is a benefit to many companies, it is also a weapon for cyberattacks. Therefore, the challenge is to find an efficient class of ML that is robustness against adversarial attacks, and that can powerfully process data.

A great option is Graph Neural Network (GNN), where information can be represented as graphs. Due to this complex and huge subject, this study will be limited to GNN as Host-IDS and Network-IDS, and how far it is robust against attacks. 

= Graph Neural Networks

The definition of the graph used in this work is as:
$
  G = (V, E)
$
Where $V$ denotes the set of nodes and $E$ the set of edges. The graph can be heterogeneous, which means the nodes can have various types, or homogenous, there is only one type of node. The graph is described by its adjacency matrix $A$ ($abs(V) times abs(V)$). Its nodes features set is represented by a matrix $X space (abs(V) times d_v)$, with $d_v$ the dimension of the nodes' feature space. It is equally possible to define $X_"egdes" space (abs(E) times d_e)$, to attach an embedding to each edge for edge classification or prediction @graph_book.

== Graph Encoding and Random Walks
Deep learning on graph networks represent a challenging task as graphs can have an arbitrary size and a complex topological structure, in contrary to an image for instance, that have a spatial locality, on a grid. Moreover, graphs often have multimodal features and lack ordering. An encoding should not be dependent on a reference point as this has no sense in graph theory. Thus, the importance to find another way to map similar nodes in the graph closely in the embedding space as shown in @fig:encoding.

#figure(
  image("resources/graph_encoding.png", width: 55%),
  caption: [The embedding should reflect the structure of the graph, i.e. $"similarity"(u, v) approx ub(Z)_u^top ub(Z)_v$ @gauthier_cgnn_2024.]
) <fig:encoding>

To achieve this, a naive algorithm to begin with is the *Random Walk Optimization*. It consists in running a short fixed-length random walk starting from each node $u$, collecting the multiset of nodes visited $v$, and optimizing the embeddings such that $Pr(v_"visited" | u_"starting point") prop ub(Z)_u^top ub(Z)_v$. This technique is still frequently used on multiple tasks like intrusion detection or adversarial attacks on graphs neural networks. It has been improved with algorithms like Deep Walk or Node2Vec.

#figure(
  image("resources/node2vec.png", width:50%),
  caption: [The Node2Vec algorithm uses biased random walks to control how fast the walk explores and leaves the neighbourhood of the starting node @node2vec.]
) <fig:node2vec>

== Graph Convolutional Network (GCN) 
To apply machine learning to node embedding, common techniques have been taken from traditional neural networks. The first one is the idea of convolutions. As convolutions collect information from the neighbours of each pixel to update the features of this same pixel, they are well suited for the task of embedding nodes. In a way, Graph Convolutional Networks are the generalization of convolutional neural network to non-euclidean spaces. At each iteration of a GCN layer, the following operation is performed on the features of each node: 
$
  ub(h)^((l+1))_u = sigma (sum_(u in cal(N)_v union {u}) 1/c_(u v) ub(W)^((l)) ub(h)^((l))_v)
$ <eq:gcn-node-update>
Where $cal(N)_v$ are the neighbours of the node $u$, $1/c_(u v)$ is a normalization coefficient, $ub(W)^((l))$ corresponds to the weight parameter to transform information from node $v$, and $sigma$ represent the non-linear activation, usually a ReLU. The embedding of each node is then updated at each GCNConv layer with $ub(h) space (ub(h)^((0)) = X)$.

#figure(
  image("resources/gcn.png", width: 70%),
  caption: [The structure of a single layer of a GCN (GNNVis @gnnvis).]
) <fig:gcn>

As computing new features node per node is unfeasible in practice, GNN and other common graph-based algorithms use operations on matrices. The previous equation @eq:gcn-node-update becomes: 

#let dtilde = $tilde(D)^(-1/2)$
$
  ub(H)^((l+1)) = sigma(dtilde tilde(A) dtilde ub(H)^((l)) ub(W)^((l)))
$
Where $tilde(A) = A + I_N$, and $tilde(D)_(i j) = sum_j tilde(A)_(i j)$. One can also define $hat(A) = dtilde tilde(A) dtilde$. A GCN with a single layer is then defined by:
$
  Z = "softmax"(hat(A) sigma(hat(A) ub(X) ub(W)^((1))) ub(W)^((2)))
$ <eq:gcn-one-layer>

== Graph Sample and Aggregate (GraphSAGE)
The generalization to GCN is the GraphSAGE structure. The general equation is given below @eq:graph-sage. 
$
  ub(h)^((l))_u = "UPDATE"(ub(h)^((l-1))_u, "AGGREGATE"({ub(h)^((l-1))_v, forall v in cal(N)(u)}))
$ <eq:graph-sage>
Where the AGGREGATE function is a generic function that could be a mean, a pooling or even a Long-Short Term Memory unit (LSTM). 

The GraphSAGE architecture is wildly implemented as it has a significant advantage: its speed. In fact, the aggregation function is unapplied on all the neighbours, but only on a fraction of them. This fraction is chosen randomly, and this downsamples allows GraphSAGE structure to be stored in dense adjacency lists, which drastically improves efficiency. For this reason, many IDS use GraphSAGE to perform detection in real time on large datasets. 

Another factor that enhance the practicability of GraphSAGE is its inductive nature. Graph learning can be either inductive or transductive. The first one is the same as supervised machine learning. However, in transductive learning both training and testing dataset is exposed to the model in the learning phase itself, only the features of the test nodes are unrevealed. As the GCN is transductive, each step of the learning algorithm needs the whole graph, which can be impracticable in a real scenario. The transductive issue will be discussed in the following sections. 

== Graph Attention Network (GAT)
GCN and GraphSAGE use aggregation functions that do not take into account the nature of the edge, the functions are often orderless, like mean or max. However, it may be useful to address each neighbour differently. Thus, the graph attention networks an attention weight $alpha_(u v)$ to the previous equation, which gives: 
$
  ub(h)^((l))_u = "UPDATE"(ub(h)^((l-1))_u, sum_(v in cal(N)(u)) alpha_(u v) ub(W)^((l)) ub(h)_v^((l-1)))
$

In GCN and GraphSAGE, this weight is equal to $1/abs(cal(N)_u)$, whereas in GAT, it is computed with a MLP layer and a softmax function. 

GAT and their counterparts, the Heterogeneous Graph Attention Network (HAN) are used for their attention mechanism as they can capture the semantic of meta-paths in the graph. This could correspond to botnet activities for instance, as we will discuss in more detail in incoming sections. 

The other GNN described in this work are based on the same techniques than the ones presented, usually changing the UPDATE or the AGGREGATION functions. A slightly different last one would be the less common Graph Isomorphism Network (GIN) @xu_powerful_2019, which is based on different assumptions, i.e. the injectivity of the involved functions, to serve a different goal: discriminative power. We still choose to mention it, as it is used in some of the presented works to bypass a number of the other GNN structures' limitations. 

= GNN-based Intrusion Detection Systems (IDS)

There are a few advantages to use GNN for Introduction Detection. Firstly,
the *structural property of data*, where data are semi-structured information. GNNs excel in discerning and learning from these structural relationships, and it is crucial for optimal detection performance. Secondly, the *higher-order interaction*. There are structural similarities between data: GNN can subsequently find nodes that exhibit similar patterns of information propagation than an infected node. Finally, the *behavioural supervisory signal*. GNN-based models can be improved by incorporating additional context.

#figure(
  image("resources/schema.png", width: 70%),
  caption: "Difference between conventional IDS and GNN-based IDS"
)

To begin our research, we first read the survey of Bilot et al. (2023) @bilot_graph_2023. This survey states that many GNN-based defences use random walk methods, but these techniques are usually challenging to apply in practice due to the inductive setting. Moreover, these methods are highly dependent on the hyperparameters and tend to prioritize proximity information above structural information. It also mentions spectral methods tend to be less used as they are inherently transductive. Thus, we will focus on more advanced GNN structures, like GCN, Heterogenous Attention Network or Hyper Graphs Neural Networks. The survey lists studies using random walk classification @bowman_detecting_2020 @ramesh_2022, GCN architecture @zhou_automating_2020 @zhang_practical_2022 @zhao_heterogenous_2020 @yongyi_denial_2022, GIN architecture @lo_xgbot_2023 @yuzhen_graphddos_2022, GraphSAGE architecture @lo_egraphsage_2022 @chang_graphbased_2021 @jin_eminbatch_2022 @caville_anomale_2022, and even RNN architectures @isaiah_euler_2023. Other studies tried capturing the semantic of attacks with meta paths @fucheng_mltracer_2020 @yong_lmtracer_2021 @xiaoqing_hetglm_2022. 

The general architecture for a GNN-based IDS is composed of three classical steps: the preprocessing, the embedding and the detection and training step. This procedure is shown in @fig:ids-3steps.
#figure(
  image("resources/ids-3steps.png"),
  caption: [General architecture for intrusion detection with GNN-based methods @bilot_graph_2023.]
) <fig:ids-3steps>

There are multiples ways to represent network data as graphs. One can construct *Packet Graph*, to deal directly with IP packets; *Flow Graph*, to capture the semantic of a whole TCP or UDP communication; or *Authentication Graph* that represent, as its name suggests, IP addresses as nodes and authentications as edges. 

Packet graph network is rarely used in literature. As the majority of the traffic is encrypted under TLS @google_report_2023, the payload analysis in unreliable, combined with the fact that creating edges for each packet creates scalability issue. In that manner, we will focus on the other two graph structures. 

There are two main classes: *Static graph*, for analysing historical and fixed data structures, and *Dynamic graph*, for real time and immediate response.

== Graph embedding: a technique to capture the semantics of attacks
Studies showed different ways of encoding network information into graphs. The most basic case is to encode systems as nodes and information flows as edges. This approach, made for instance by @zhou_automating_2020, can then be used to apply GCN to detect malicious nodes. However, such a raw encoding lacks information, like long-term dependences traditional GCN are unable to catch due to the over smoothing problem @graph_book. In point of fact, as the diameter of a graph tends to be low compared to the number of nodes (usually six in social networks), and as a $m$-layers GCN aggregates information from the $m$-hop neighbours, information is diluted to the point of uselessness when the number of layers is too significant @zhou_automating_2020 @zhang_practical_2022. This is a reason why using only GNN layers may be unappropriate. 

To assess this issue, Lo developed XG-BoT @lo_xgbot_2023, a framework to capture the long dependences before using GNN layers. As shown in @fig:xg-bot, each node's features go through a GIN network before being used in the node embedding part. 

#figure(
  image("resources/xg_bot.png"),
  caption: [XG-BoT architecture: using GIN before conventional GNN layers to overcome the oversmoothing issue.]
) <fig:xg-bot>

As we shall see in the following examples, GNN-based IDS do not only use on type of GNN to perform detection, it remains purely a part of the system. The rest is frequently achieved by other ML architectures, or special pre/post-processing. For instance, some studies tried meta-paths as a preprocessing technique @zhao_heterogenous_2020 @fucheng_mltracer_2020 @yong_lmtracer_2021 @xiaoqing_hetglm_2022. The goal is to handcraft common paths involving multiple nodes and edges, that the GNN model will then be able to recognize and generalize. 

#figure(
  image("resources/meta_path.jpg"),
  caption: [An heterogenous graph with meta-path preprocessing @zhao_heterogenous_2020.]
) <fig:meta-path>

Meta-paths are frequently used as they offer explainability and a low rate of false negatives @fucheng_mltracer_2020. They can be used in other scenario, like intrusion detection on a single machine. The graph is then a provenance graph from processes inside the machine. An example can be found with the study @lv_heterogeneous_2021. This study chooses to use GAT and graph sampling to drastically reduce the size of sub-graphs. 

Flows are not always represented as edges, some studies used nodes for systems and nodes for edges @pujol-perich_unveiling_2021, we show their results in @sec:against-trad.

As we mentionned earlier, the GraphSAGE architecture is popular for detection intrusion tasks, as it can scale on large graphs. For example, the Anomal-E @caville_anomale_2022 frameworks is based on this GraphSAGE architecture and implements a self-supervised way to detect intrusions. It is interesting as it relies on both graph methods and traditional anomaly detection methods. It also does not need any label, as the encoding part is done with the Deep graph infomax method @velickovic_deep_2018, which is an unsupervised one, this makes the approach very practical to real-world scenarios. Hence in this case, the graph structure is only used to aggregate information. 

A study @fu_detecting_2023 (@ex:hypervision) describes the use of real-time detection with an unsupervised graph. This approach describes a graph construction on flow size classification. In this case, preprocessing allows reducing density and then identify malicious vertices with a loss function. The interesting points are the three metrics which allow evaluating the information retained by exiting traffic: the amount of information, with Shannon entropy; the scale of data; the density of information.

#figure(
  image("resources/hypervision.png", width: 90%),
  caption: [The structure of HyperVision: subgraph are constructed using flow interactions (@ex:hypervision).]
)

#figure(
  image("resources/hypervision_examples.png", width: 50%),
  caption: [Example of subgraphs for common attacks (@ex:hypervision).]
)

For real-time detection, dynamics graphs needs Flow that are modelled as a sequence of a per-packet feature represented by an aperiodic irreducible discrete-time Markov chain.

== Graph detection against traditional detection <sec:against-trad>
Pujol-Perich et al. @pujol-perich_unveiling_2021 analysed the advantage of using GNN-based IDS over traditional IDS. They use nodes to encode systems and flows, but their approach is similar to the previous ones. They compared basic IDS techniques, like Random Forests or ID3 decision tree to their method. The table @table:trad-vs-gnn resumes the results. 

#figure(
  table(
    stroke: none,
    columns: 6,
    table.vline(stroke: 0.05em, x: 1),
    [*Class label*], [*MLP*], [*AdaBoost*], [*RF*], [*ID3*], [*GNN*],
    table.hline(stroke: 0.05em),
    [Benign], [0.67], [0.68], [*0.99*], [*0.99*], [*0.99*],
    [SSH-Patator], [0.0], [0.0], [*0.99*], [*0.99*], [0.98],
    [FTP-Patator], [0.0], [0.0], [*0.99*], [*0.99*], [*0.99*],
    [DoS GoldenEye], [0.12], [0.0], [0.97], [0.96], [*0.99*],
    [DoSHulk], [0.63], [0.63], [*0.99*], [*0.99*], [*0.99*],
    [DoS Slowhttptest], [0.01], [0.0], [*0.98*], [*0.98*], [0.97],
    [DDoS], [0.51], [0.0], [*0.99*], [*0.99*], [*0.99*],
    [Web-Brute Force], [0.0], [0.0], [*0.82*], [0.76], [0.73],
    [Web-XSS], [0.0], [0.0], [0.69], [0.65], [*0.83*],
    [Bot], [0.0], [0.0], [*0.98*], [*0.98*], [*0.98*],
    [Port Scan], [0.78], [0.0], [*0.99*], [*0.99*], [*0.99*],
  ),  
  caption: [Accuracy of different ML-based NIDS over the CICIDS2017 dataset.],
  kind: table,
) <table:trad-vs-gnn>

But the authors mention that the real value of GNN are their robustness to perturbations. As shown in the @fig:gnn-ids-robust.

#figure(
  image("resources/gnn-ids-robust.png", width: 50%), 
  caption: [Evaluation of the accuracy of ML-based NIDS under variations of the packet size (top) and the inter-arrival time (bottom) @pujol-perich_unveiling_2021.] 
) <fig:gnn-ids-robust>

However, even if GNN may handle normal perturbations, we may see in the next section that they are vulnerable to adversarial perturbations. 

= Robustness of Graph Neural Networks
Deep learning models on graphs excel in tasks like node classification, link prediction, and graph clustering. However, they reveal vulnerabilities to adversarial examples, making them uncertain and unreliable. Given their use in IDS, ensuring the robustness of GNNs is crucial. A typical attack scenario would represent an attacker that compromised a node, i.e. a system, who tries to perform additional benign actions that are useless for the attack. However that compromise the GNN based IDS, thus making the attack stealthy. Put differently, if an IDS detects the attack and classifies the targeted node as compromised, the attacker wants to perform new actions to change the prediction of the IDS for the targeted node. This attack is depicted in the @fig:adding-fake-edges.

#figure(
    image("resources/adding_fake_edges.png", width: 55%),
    caption: [A misclassification of the target caused by small perturbations of the graph structure and node features. @chen_survey_2022]
) <fig:adding-fake-edges>

This attack scenario can be formalized: given $cal(T) subset V$ the target set of nodes, the attacker aims to find a perturbation in the graph, $hat(G) = G + Delta G$, such that the new graph $hat(G)$ satisfies: 
$
  &op(max, limits: #true)_(hat(G) in Psi(G)) sum_(t_i in cal(T)) cal(L)(f_(theta^*)(hat(G), X, t_i), y_i) \
  &"s.t." theta^* = op("argmin", limits: #true)_theta sum_(v_i in V) cal(L)(f_(theta^*)(hat(G), X, v_i), y_i)
$ <eq:general-attack>

To make the attack as imperceptible as possible, the attacker also uses a comparison function $cal(Q)$, such that:
$
  &cal(Q)(hat(G), G) < epsilon \
  &"s.t." hat(G) in Psi(G)
$
Where $Psi$ defines the perturbation space of attacks, $f$ the deep learning model with its set of parameters $theta$, and the loss function $cal(L)$ @chen_survey_2022.

== Attacker's Knowledge 
The attack can be a *White-box Attack*, a *Gray-box Attack* or a *Black-box attack*. As white-box attacks are impracticable in the real world, many papers focus on the grey-box attack, which implies knowledge only on the targeted nodes. This fits a scenario where the attacker has access to the target node and tries to become stealthy by changing the target node's prediction. 

A new type of attack has emerged and has become widely used because of its practicability: the *No-box Attack*. The principle is to create a surrogate model based on the limited understanding of the target model. The attack is then performed on the surrogate model as a white-box attack. Its success will depend on the transferability of the surrogate model. 

The attacker can have a *Perfect Knowledge*, *Moderate Knowledge* or a *Minimal Knowledge*. Even if the first case is impracticable, it is the most common case used in previous studies (2022). 

== Types of attacks
Contrary to other machine learning fields where attacks such as model extraction or membership inference attack exist, attacks on graph neural networks remain focused on two categories: *Poisoning Attacks* and *Evasion Attacks*. The popularity of the first one can be explained by the wild use of transductive learning which requires test samples, but not their label, during the training stage. 

While poisoning attacks can be defined according to the general equation @eq:general-attack. Evasion attacks occur after the target model is well trained on a clean graph. 

== Attack Strategy
The attack can be either a *Topology Attack* that targets the adjacency matrix by adding or removing edges. Or a *Feature Attack*, which alter the features of the targeted nodes. In both cases, a budget $Delta$ can be defined:

$
  underbrace(sum_(u, i) abs(ub(X)_(u i)^((0)) - ub(X)_(u i)^'), "Feature budget") + underbrace(sum_(u < v) abs(A_(u v)^((0)) - A_(u v)^'), "Topology budget") <= Delta
$

== Taxonomies of Attacks
There are plenty of techniques used to perturbate the GNN's predictions. Some of them are specific to graph learning, such as random walk attacks; other ones come from different machine learning fields, like the gradient-based attacks. 

The survey of Chen et al. @chen_survey_2022 gave a list of such techniques (2022). There are gradient-based algorithms, such as the iterative gradient attack with an auto-encoder framework by @chen_link_2018 or @sun_poisoning_2028, reinforcement learning based attack methods with @ma_attacking_2019 or @sun_survey_2022, random walk or GCN as surrogate model to attack general GNN @bojchevski_adversarial_2019 @dai_adversarial_2018 @chen_fast_2018 @zugner_adversarial_2018 @wang_attacking_2019, meta-gradients attacks @zugner_adversarial_2024 (2024), generative adversarial network attacks @wang_attack_2020, algorithms based on deepwalk, eigen decomposition and genetic algorithm @bojchevski_adversarial_2019 @yu_unsupervised_2019, auto-encoder based algorithms @bose_generalizable_2020, and heuristic algorithm @waniek_attack_2018.

=== Use of surrogate models
As mentioned, some studies used simpler model to perform their attack. In the case of a GCN, a surrogate model can be produced by removing the non-linearities @zugner_adversarial_2018, i.e. the activation functions. The surrogate model then become: 
$
  Z' = "softmax"(hat(A) hat(A) ub(X) ub(W)^((1)) ub(W)^((2))) = "softmax"(hat(A)^2 ub(X) ub(W))
$
This method is described in the appendix @ex:linear-gcn-surrogate. Another study uses the LinLBP method @wang_attacking_2019 as a surrogate model. The equation then becomes: 
$
  p = q + A dot.circle W p 
$

With $p$ and $q$ the evolving embedding. The adversarial edges are then transferred to be used on GCN. The method is explained in @ex:linlbp.  

=== Finding techniques to ease the computation
To determine if adding or removing an edge can perturbate a system, it is necessary to compute the new adjacency matrix, which can be computationally intensive. This is why studies tried to find approximations or new ways of computing the adjacency matrix. For example, in Zügner's paper @zugner_adversarial_2018, they found the new adjacency matrix could be written as:

$
  [hat(A)'^2]_(u v) = [hat(A)^2]_(u v) + Delta_(u, v, m, n)
$
With $Delta_(u, v, m, n)$ computed in $O(1)$. 

Bojchevski et al. @bojchevski_adversarial_2019 found a way to manipulate easily the spectrum of the graph. This is done by computing the potential change in the eigen values when adding or removing an edge:
$
  Delta lambda_y = Delta w_(i j)(2u_(y i)u_(y j) - lambda_y (u_(y i)^2 + u_(y j)^2))
$
As mentioned in @ex:spectrum, spectrum based detection can be bypassed using this technique. 

=== Current Limitations
The two main limitations are the *Unnoticeability* and the *Scalability*. Even if studies have often defined an attack budget $Delta$, it is lacking in semantics. As mentionned in @zugner_adversarial_2018, the graph's statistics, such as the degree low, should remain intact, thus the attack should preserve the power distribution law. The second limitation is not limited on graphs, but it can be even more problematic in the field of Graph Neural Networks, as many attacks require the whole graph to perform. 

High-degree nodes are much harder to attack than low-degree nodes. Thus, the $Delta$ chosen in studies like @zugner_adversarial_2018 @bojchevski_adversarial_2019 is linear in the degree of the node, i.e. $Delta = d_v_0 + a$, ($a = 2$ in @zugner_adversarial_2018 and $a = 3$ in @bojchevski_adversarial_2019). 

Most of the studies use the Cora dataset or social networks, thus it is yet difficult to know if attacks can be efficient on more complicated datasets, especially on IDS.


== Taxonomies of Defenses
Currently, the amount of research for defense is far less that that of attack on the graph domain. Moreover, as Chen's survey @chen_survey_2022 mentions it, most existing works (2022) for defense only focus on node classification tasks. It cites a list of used techniques. There are techniques based on preprocessing, like to drop special edges that could be risky @wu_adversarial_2019. Techniques that change the structure of the traditional GNN to improve their robustness, such as the Gaussian-based graph convolutional layers @zhu_robust_2019, or the Adaptive Graph Convolutional Network @ioannidis_edge_2019. Other techniques change the training process, for instance by implying adversarial training @xu_topology_2019 @sun_virtual_2020 @feng_graph_2019 @chen_adversarial_2019 @deng_batch_2019 @he_adversarial_2018. This is a technique already well known in other fields of machine learning, or by changing the loss function to a more robust one @zugner_certifiable_2019 @bojchevski_certifiable_2019 @jin_power_2021 @chen_adversarial_2019. Some tried applying a protection on top of the existing model to try to detect adversaries @zhang_comparing_2019 @ioannidis_graphsac_2019 @hou_alphacyber_2019. Finally, some changed the optimization and the aggregation process to prevent transferability attacks @tang_transferring_2020.

= Conclusion

In this paper, we present Graph Neural Network and the interest to use it as GNN-based IDS. Instead of conventional IDS, this one use graphs to represent data like address and hosts. The interest is that GNN excel in discerning relationships between nodes, but it can also perform higher-order interaction and can be improved by adding additional content. Specially, this leads to have static graph for analyse data structures and dynamic graph for real-time detection.

Moreover, graph-based detection is highly practical with the right framework, as it has been proven to be faster and more energy efficient than other ML based techniques @protogerou_graph_2021. 

Concerning adversarial attacks, there are not enough studies. However, it appears to be possible to generate adversarial perturbation that could make the attacker invisible, even if GNN seem more robust than other traditional ML models.

#pagebreak()

= Appendix

== Taxonomies summary

A summarized of GNN based IDS. It includes Graph Construction and Attacks against them. About construction, GNN can be used for Host Intrusion Detection, Network Intrusion Detection or even Controller Area Network Intrusion Detection.

#figure(image("resources/color.png", width: 120%))

== Example of a linear GCN as surrogate model to find adversarial perturbations <ex:linear-gcn-surrogate>
Zügner et al. @zugner_adversarial_2018 proposed in 2018 an attacked based on a surrogate model. The principle is straight forward, linearize as much as possible the model's equation to be able to solve the minimisation problem given constraints. 

Given a GCN model defined as before @eq:gcn-one-layer, the surrogate model is created by removing the non-linearities, i.e. the $sigma$ function:
$
  Z' = "softmax"(hat(A) hat(A) ub(X) ub(W)^((1)) ub(W)^((2))) = "softmax"(hat(A)^2 ub(X) ub(W))
$
The softmax can also be removed when computing the log-probabilities, thus leaving only one non-linearity: the square. Hence, it is still challenging to attack the model by adding or removing edges. To solve that problem, the authors found a way to fast compute the new adjacency matrix squared $hat(A)^2$. With $A'$ the new adjacency matrix obtained by adding or removing the edge $e = (m, n)$, $hat(A)'^2$ can be computed with: 
$
[hat(A)'^2]_(u v) = [hat(A)^2]_(u v) + Delta_(u, v, m, n)
$
With $Delta_(u, v, m, n)$ computed in $O(1)$. Nonetheless, a feature attack is easier. To find the best node and feature  $(u^*, i^*)$ one only needs to compute the gradient: 
$
  Gamma_(u i) = [hat(A)^2]_(v_0 u) ([ub(W)]_(i c) - [ub(W)]_(i c_"old"))
$
With $v_0$ the target node, $c_"old"$ the old class and $c$ the new class. This is the perfect example to demonstrate how to transform the initial problem into a gradient-based one. However, this approach is limited to simple architectures of GNN, due to transferability issues. It also requires a full knowledge of the features $ub(X)$ which may be impossible in practice. 

== Example of a Linearized Loopy Belief Propagation model as surrogate model <ex:linlbp>
The paper of Wang et al. @wang_attacking_2019 uses a similar technique to @zugner_adversarial_2018: find a simple surrogate model to use gradient-based techniques. However, they went a bit further as they made the adjacency matrix become continuous, a technique that can be used in other scenario. 

The targeted model is the Linearized Loopy Belief Propagation (LinLBP). Suppose we are given an undirected graph, a training dataset $L$ which consists of a set of labeled positive nodes $L_P$ and a set of labeled negative nodes $L_N$. LinLBP assigns the prior reputation score $q_u$ for a node $u$ as follows: 
$
  q_u = cases(
    theta &" if" u in L_P,
    -theta &" if" u in L_N,
    0 &" otherwise"
  )
$
where $0 < theta <= 1$ is a parameter of LinLBP. LinLBP assigns the same weight $w$ in the interval (0, 0.5] for all edges. A larger weight means that two linked nodes are more likely to have the same label. We denote by $W$ the weight matrix ($abs(V) times abs(V)$), every entry of which is the weight $w$. Then, the posterior reputation scores in LinLBP are a solution of the following system: 
$
  p = q + A dot.circle W p 
$
where $q$ and $p$ are the column vector of prior reputation scores and posterior reputation scores of all nodes. To solve this equation, the scores are iteratively computed as follows: 
$
  p^((t)) = q + A dot.circle W p^((t-1))
$
where $p^((t))$ is the column vector of posterior reputation scores in the $t$th iteration. When the posterior reputation scores converge, a node $u$ is predicted to be negative if its posterior reputation score is negative, i.e., $p_u < 0$. We note LinLBP has two parameters, the prior reputation score parameter $theta$ and the edge weight $w$. 

The attack then consists of finding $B$ such as:
$
  op(min, limits:#true)_B&  sum_(u in S, v in V') B_(u v) C_(u v) + n C_"node" \ \ \ 

  "s.t." &"FNR" = 1, "for the fake nodes" \
  &"FNR" = 1, "for the target nodes" \ 
  &B_(u v) in {0, 1}, u in S, v in V' \
  &sum_v B_(u v) <= K, u in S 
$
With:
- $B$, a binary matrix that indicates a change by the attacker. If $B_(u v) = 1$, it means the edge  $(u, v)$ has been either removed if it existed before, or added if it did not. 
- $C_(u v)$, the cost of insterting an edge between nodes u and v. 
- $C_"node"$, the cost of creating each fake node. 
- $n$, the number of fake nodes. 
- $K$, the max number of edges each fake node can have. 
- $"FNR" = 1$, achieve as high false negative rate as possible. 

This attack is then viewed as an optimisation attac. Using Lagrangian multipliers it is possible to achieve this attack in seconds on large datasets where the Nettack attack took hours to compute. It is possible because $B$ is not treated as binary but as a continuous matrix during the computation, and then converted back into a binary matrix. This allows the attacker to use common gradient based algorithms like a projected gradient descent. 

This attacks works well on collective classifications methods, like LinLBP, with a success of over $0.92$ FNR. However, it can be transferred to Graph Neural Network based classification methods like CNN where it achieves $0.54$ FNR. 

== Example of a spectrum-based attack on unsupervised learning <ex:spectrum>
The paper from Bojchevski et al. @bojchevski_adversarial_2019 shows how to find adversarial edges to modify the spectrum of the graph. The spectrum of the adjacency matrix is often used in graph analysis as it reflects many properties of the graph. It is also used in intrusion detection @majed_spectral_2024.

The idea of this paper is to exploit gradient based methods on the spectrum of the adjacency matrix leveraging eigenvalue perturbation theory. To compute the attack efficiently, it is necessary to have an efficient way to computing the new eigen values for each edge flip. Let $u_y$ be the $y$th generalized eigenvector of $A$ with the generalized eigenvalue $lambda_y$. Then the generalized eigenvalue $lambda_y^'$ of $A'$ is approximatively $lambda_y^' = lambda_y + Delta lambda_y$, with $Delta lambda_y$ defined as: 
$
  Delta lambda_y = Delta w_(i j)(2u_(y i)u_(y j) - lambda_y (u_(y i)^2 + u_(y j)^2))
$
This equation allows changes in $O(1)$, thus enabling the attack. 

However, this approach is mainly used to poison deep walk methods. The results show that a big number of edges have to be flipped in order to have an influence on the predictions. The results are shown in @fig:dw-poisoning.
#figure(
  image("resources/dw_poisoning.png"),
  caption: [Targeted attack on the link prediction task for Cora and Citeseer datasets.]
) <fig:dw-poisoning>

Nonetheless, an interesting discovery has been made by the authors: the adversarial edges found with this method do not appear to have special traits like high centrality. Thus, they concluded classical methods can not be used to find those edges. 

== Example of more advanced graph structures for intrusion detection: Hyper Graph Neural Networks <ex:hgnn>

The main idea behind the use of hypergraphs is that the original graph structure primarily captures pairwise node correlations, with limited capability to model complex, higher-order interactions among multiple nodes. As mentioned in @yang_hrnn_2024, hypergraph structures can effectively model and express complex relationships among multiple nodes, this capability is in theory, beneficial in extracting intricate traffic patterns and relational networks. Another problem is that, usually, the spatial and temporal informations are separated, thus the distinct attention mechanisms can suppress critical information, needed by the other part. 

The hypergraph is constructed as follows: each flow is a vertex; a hyperedge is created between two similar nodes (based on an euclidean distance in an unknown space), then convolutions are used to encode high-order relationships between flows. This procedure is shown in @fig:hrnn-construction and @fig:hrnn-module.
#figure(
  image("resources/hrnn_construction.png"),
  caption: [The schematic of hypergraph construction.]
) <fig:hrnn-construction>
#figure(
  image("resources/hrnn_module.png"),
  caption: [A HRNN Module.]
) <fig:hrnn-module>

The authors conclude HRNN generates both robust and semantically expressive embeddings, significantly improving the detection capability of anomaly traffic. This leads to state-of-the-art results on famous datasets like NSL-KDD, ISCXVPN2016 or even CIC-IDS2017. 

Even if this approach seems interesting, no code is provided and the paper lacks crucial information on why the construction of hypergraph is able to grasp the semantic of attacks. 

== Example of different meta-paths <ex:meta-path>
#figure(
  image("resources/meta_path_def.jpg"),
  caption: [Meta-paths in @zhao_heterogenous_2020.]
)

== Example of HyperVision, a real time unsupervised ML <ex:hypervision>

Since 2019, 80 percents of websites has used HTTPS protocol.
#link("https://github.com/fuchuanpu/HyperVision")[HyperVision] is a new method to achieve unsupervised detection -- more realistic case -- and it is able to detect encrypted malicious traffic without accessing to the payload. The interesting point is it needs to build a graph for realtime detection.

Usually, flows on Internet are short meanwhile most packets are long. The tool HyperVision records interaction patterns to reduce density of the graph. This one is build with addresses as vertices and flows as edges. To aggregate short flows, the flows must:
- have the same source and/or destination;
- have same protocol;
- be repetitive enough.

For connectivity analysis, connected components are found by using depth-first search. This leads to extract abnormal components.

To detect malicious traffic, critical vertices are identified with solving the vertex cover problem: in a component, each flows with a source and/or destination of a malicious flow are grouped in a subset. Those edges connected to a critical vertex are clustered, and K-Means algorithm is used to calculate the loss function that indicates the degree of maliciousness.

There are three equations: $text("loss"_"center")$ that indicates the difference from other edges connected to the critical vertex; $text("loss"_"count")$ that is the number of flows denoted by the edge; and finally $text("loss")$ function that when it became greater than a threshold indicates maliciousness of the edge.
$
"loss"_"center" ("edge") = op(min, limits: #true)_(C_i in {C_1,...,C_k}) norm(C_i - f("edge"))_2 \
"loss"_"count" ("edge") = log_2("Size"(C("edge"))+1) \
"loss"_"cluster" ("edge") = "TimeRange" (C ("edge")) \
"loss" ("edge") = alpha "loss"_"center"("edge") - beta "loss"_"cluster" ("edge") + gamma "loss"_"count" ("edge")
$

Where K is the number of cluster centers, $C_i$ is the $i^"th"$ center and $f("edge")$ is the feature vector.

#pagebreak()