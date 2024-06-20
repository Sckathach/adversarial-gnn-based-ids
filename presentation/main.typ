#import "@preview/polylux:0.3.1": * 
#import themes.simple: * 

#set text(font: "Inria Sans")
#show: simple-theme.with(
  footer: [SSR Presentation 2024, Romain Moreau & Thomas Winninger]
)

#show figure: X => [
  #set align(center)
  #X 
]


#title-slide[
  = Graph Neural Network based Intrusion Detection and its Robustness against Adversarial Attacks
  #v(1em)
  Romain Moreau #footnote[Students - Télécom SudParis] <fn> #h(1em)
  Thomas Winninger @fn
  
  Gregory Blanc #footnote[Supervisor - Télécom SudParis]

  June 17, 2024

]

#slide[
  = Intrusion Detection System
  #figure(image("assets/ids.png")) 
]

#slide[
  = Graph Neural Network
  #figure(image("assets/node2vec.png", width: 60%))
]

#slide[
  #figure(image("assets/gcn.png")) 
]

#slide[
  = GNN-based IDS
  #figure(image("assets/ids_graph.png", width: 60%))
]
#slide[
  #v(2em)
  #figure(image("assets/ids_system.png", width: 70%))
]

#slide[
  #v(2em)
  #figure(image("assets/meta_path.jpg", width: 100%))
]
#slide[
 #figure(image("assets/hypervision.png")) 
]

#slide[
  #figure(image("assets/hypervision_examples.png"))
]


#slide[
  = GNN adversarial attacks
  #figure(image("assets/adding_fake_edges.png", width: 60%))
]

#centered-slide[
  = Conclusion
]