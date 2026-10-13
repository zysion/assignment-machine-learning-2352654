The diferent between NB and TAN is:

NB:
- each feature in a class are independent
- diagram structure of NB is one class node connect to all feature node; no have any edge between feature node.
- each feature just only have 1 parent node that is node class.
- $P(X\Vert{}Y)$ = $\prod_{i=1}^n P(X_i \vert{} Y)$

TAN:
- a feature can have relationship with the other feature in class.
- diagram struct of TAN include NB but have some edge to describe the relationship between each feature.
- eachs feature can have more than 1 parent.
- $\prod_{i=1}^n P(X_i \vert{} Y, \text{Parent}(X_i))$