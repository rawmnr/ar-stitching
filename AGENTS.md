# AGENTS.md

## Mission actuelle
Nous sommes dans la phase de fondation du projet.
Ne pas optimiser encore l’algorithme de stitching.
Objectif: construire un laboratoire de simulation robuste, falsifiable, testable.

## Règles absolues
- Ne jamais modifier `src/stitching/trusted/**` sans aussi ajouter/mettre à jour des tests.
- Ne jamais réduire la validation à une seule métrique RMS.
- Les métriques géométriques et de footprint sont bloquantes.
- Ne jamais proposer une solution qui masque des pixels pour améliorer artificiellement le score.
- Ne jamais lisser massivement une reconstruction juste pour faire baisser le RMS.
- Toute nouvelle primitive doit venir avec:
  - tests unitaires
  - au moins un test de propriété ou métamorphique
  - documentation courte
- Privilégier fonctions pures, typage, dataclasses/Pydantic si utile, NumPy/SciPy vectorisé.

## Architecture
- `src/stitching/trusted/`: génération de vérité terrain, extraction des sous-pupilles, bruit, biais instrument, métriques, règles d’acceptation.
- `src/stitching/editable/`: futures implémentations de stitching modifiables par agent.
- `src/stitching/harness/`: orchestration, exécution, rapports.

## Ce qui doit être vrai avant toute optimisation
- génération de surfaces testée
- génération de footprint testée
- translations/rotations testées
- biais instrument stationnaire dans le repère capteur
- bruit et outliers injectés de manière contrôlée
- scénarios canoniques versionnés
- évaluation combinant géométrie + signal

## Métriques minimales
- footprint_iou
- valid_pixel_recall
- valid_pixel_precision
- largest_component_ratio
- hole_ratio
- rms_on_valid_intersection
- mae_on_valid_intersection
- hf_retention / edge preservation proxy
- runtime_sec

## Politique d’édition
- Pour cette phase, ne modifier que le scaffolding, le simulateur, l’évaluation et les tests.
- Ne pas démarrer les algorithmes CS/SC/GLS complets avant que les scénarios `s00` à `s03` soient verts.

## Workflow
- Commencer par proposer un plan court.
- Ensuite créer les fichiers minimums.
- Ensuite créer les contrats de données.
- Ensuite implémenter le scénario identité.
- Ensuite écrire les tests.
- Ensuite seulement étendre.