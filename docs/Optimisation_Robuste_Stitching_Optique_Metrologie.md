# Optimisation des Algorithmes de Stitching en Métrologie Optique : Robustesse face aux Perturbations Complexes

## Introduction à la Métrologie par Sous-Ouvertures et aux Défis de Robustesse

La métrologie optique constitue la pierre angulaire de la fabrication avancée, dictant la capacité de l'industrie à produire des composants optiques de très haute précision, tels que les asphères, les surfaces de forme libre (freeform) et les miroirs segmentés pour les télescopes spatiaux et la lithographie extrême ultraviolette (EUV).

Face aux limitations physiques des interféromètres monolithiques, notamment en matière d'ouverture numérique et de résolution spatiale, l'interférométrie par assemblage de sous-ouvertures (Subaperture Stitching Interferometry - SSI) s'est imposée comme la méthode de référence absolue.

Ce procédé repose sur le principe de l'acquisition séquentielle de multiples cartographies de phase à haute résolution sur des zones locales se chevauchant, suivie d'une reconstruction algorithmique globale permettant d'obtenir la topographie complète de la pièce avec une précision sub-nanométrique.

### Le Scénario s17

Le déploiement de la technique SSI dans des scénarios extrêmes, désignés sous l'archétype "scénario s17", révèle des vulnérabilités critiques inhérentes aux approches mathématiques traditionnelles.

Un scénario s17 se caractérise par un environnement de mesure hautement perturbé et non idéal, cumulant simultanément plusieurs sources de dégradation du signal :

- **Pentes ou courbures complexes** générant des erreurs de retour (retrace errors) sévères en condition de test non nulle
- **Instabilités thermiques** induisant des dérives lentes de la cavité interférométrique
- **Systèmes de positionnement mécaniques** introduisant des biais systématiques et du jitter lors du balayage
- **Bruits complexes** incluant desondulations de fréquences spatiales moyennes (MSF) et des valeurs aberrantes (outliers) causées par des poussières ou des diffractions aux bords de la pièce

---

## Fondements Mathématiques de l'Assemblage et Optimisation Globale

### Modèle Mathématique Sous-Jacent

Dans un maillage de sous-ouvertures, la topographie mesurée de la i-ème sous-ouverture M_i(x, y) est une superposition de :

- La topographie réelle de la surface W(x, y)
- Les erreurs de positionnement cinématique de la sous-ouverture ΔP_i(x, y)
- L'erreur systématique de référence de l'interféromètre R(x, y) (incluant les aberrations du système d'imagerie)
- Un terme de bruit complexe ε_i(x, y) dépendant de l'environnement et du capteur

Le but de l'algorithme d'assemblage est d'extraire W(x,y) avec la plus grande fidélité possible, malgré la présence de toutes les autres variables inconnues.

### Historique : De l'Assemblage Séquentiel aux Moindres Carrés Globaux

**Stitching Séquentiel** : Chaque sous-ouverture était ajustée par rapport à la précédente (1→2→3...). Cette méthode favorisait une accumulation drastique des erreurs de positionnement et de bruit, menant à des distorsions macroscopiques en forme de parapluie ou de selle de cheval sur les grandes ouvertures (phénomène connu sous le nom de "walk-off").

**Moindres Carrés Globaux (GLS)** : Toutes les sous-ouvertures sont ajustées simultanément en considérant l'ensemble du réseau de chevauchement. L'objectif est de minimiser l'erreur quadratique sur toutes les zones de chevauchement.

Cette formulation quadratique est extrêmement vulnérable aux perturbations décrites dans le scénario s17. La présence d'une erreur ponctuelle massive dans la mesure M_i (due à une poussière, une rayure, ou un artefact de diffraction) est amplifiée de manière quadratique, fausses l'estimation des coefficients de positionnement et forçant l'algorithme à incliner artificiellement les sous-ouvertures saines.

---

## Stratégies de Stitching - Comparatif

| Stratégie | Mécanisme | Avantages | Vulnérabilités (s17) |
|-----------|-----------|-----------|---------------------|
| **Stitching Séquentiel** | Minimisation itérative i→i-1 | Faible coût computationnel, implémentation triviale | Accumulation d'erreurs quadratiques, propagation sévère de la dérive |
| **Moindres Carrés Globaux (GLS)** | Minimisation matricielle simultanée via SVD/QR | Répartition homogène de l'erreur | Haute sensibilité aux outliers, conditionnement instable si faible chevauchement |
| **Optimisation Alternée (SIAC)** | Séparation stricte variables de position / forme | Découple les erreurs corrélées, stabilise la convergence | Exige critères d'arrêt stricts |
| **Optimisation Stochastique (PSO)** | Exploration globale par essaim particulaire | Immunité relative aux minima locaux | Temps de calcul prohibitif pour mégapixels |

---

## Calibration Instrumentale et Modélisation de l'Erreur de Référence

Le postulat classique selon lequel l'interféromètre fournit une onde de référence localement parfaite est totalement invalide dans les environnements de métrologie de très haute précision.

La topographie reconstruite au sein de chaque sous-verture hérite directement des défauts de topographie de l'onde de référence R(x, y), générée par un miroir plat de transmission (Transmission Flat - TF) ou une sphère de transmission (Transmission Sphere - TS), ainsi que des distorsions du système d'imagerie interne.

### Algorithmes SCS (Simultaneous Calibration and Stitching)

Le concept de calibration croisée, désigné sous l'acronyme SCS, abolit la nécessité de calibrer l'instrument hors-ligne, permettant à l'algorithme d'apprendre l'erreur du système directement à partir des données de mesure de la pièce.

---

## Estimateurs M-Robustes pour la Gestion des Bruits et Outliers

La présence de valeurs aberrantes (outliers) dans les données de mesure est l'une des challenges majeurs du scénario s17. Les estimateurs M-robustes (Huber, Tukey, Hampel) permettent de réduire drastiquement l'influence de ces anomalies sur le résultat final.

### Principe

Au lieu de minimiser la somme des carrés des résidus ( Least Squares), on minimise une fonction ρ(e) qui croît moins rapidement pour les grands résidus :

- **Huber** : ρ(e) = e²/2 pour |e|≤δ, δ|e|-δ²/2 pour |e|>δ
- **Tukey** : ρ(e) = δ²/6 pour |e|≤δ, sinon constant
- **Hampel** : Fonction piecewise avec seuils progressifs

### Implémentation Pratique

L'optimisation avec ces fonctions de perte se fait via IRLS (Iteratively Reweighted Least Squares) :

```
Pour itération = 1 à max_iterations :
    Calculer les poids w_i = ψ(r_i/δ) / r_i
    Résoudre le problème pondéré : min Σ w_i * r_i²
    Vérifier convergence
```

---

## Compensation des Dérives Spatio-Temporelles

Les dérives thermiques et mécaniques créent des variations temporelles des paramètres de mesure qui doivent être compensées.

### Approches

1. **Modélisation paramétrique** : Dépendance linéaire ou quadratique en temps
2. **Interpolation polynomiale** : Compensation par surfaces de dérive
3. **Filtrage adaptatif** : Estimation récursive type Kalman

---

## Modulation Dynamique du Front d'Onde

Pour les pièces à forte pente, la modulation dynamique du front d'onde (conjugaison de phase) permet de réduire les erreurs de retrace.

---

## Conclusions et Perspectives

La résolution du scénario s17 nécessite une approche multi-facettes combinant :

1. **Calibration automatique** des erreurs instrumentales in-situ
2. **Estimation robuste** des paramètres de positionnement via M-estimateurs
3. **Compensation des dérives** spatio-temporelles
4. **Validation croisée** sur les zones de chevauchement multiples

L'intégration de ces techniques dans un framework d'optimisation hybride représente l'état de l'art pour la métrologie des surfaces de nouvelle génération.
