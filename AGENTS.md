# AGENTS.md — Autoresearch Optical Stitching

## Mission
Optimiser itérativement l'algorithme de stitching optique dans
`src/stitching/editable/candidate_current.py` en minimisant l'erreur RMS
agrégée sur la suite de scénarios, tout en respectant les garde-fous
géométriques et temporels.

## Chemins éditables (SEULS fichiers modifiables)
- `src/stitching/editable/candidate_current.py`
- `src/stitching/editable/hybrids/*.py`

## Chemins interdits (JAMAIS modifier)
- `src/stitching/trusted/**`
- `src/stitching/harness/**`
- `scenarios/**`
- `tests/**`
- `configs/**`
- `AGENTS.md`

## Métrique principale
- **RMS agrégé** = √(mean(rms²)) sur tous les scénarios

## Métriques secondaires (surveillées, garde-fous)

| Métrique | Valeur cible | Mesure |
|---|---|---|
| `footprint_iou` | ≥ 0.99 | Intersection‑over‑Union de la zone observée |
| `valid_pixel_recall` | ≥ 0.99 | Pixel recall valide |
| `valid_pixel_precision` | ≥ 0.99 | Pixel precision valide |
| `hf_retention` | stable ou croît | Taux de rétention de haute fréquence |
| `runtime_sec` | < 300 | Temps d’exécution par scénario |
| `mismatch_rms` | cohérence interne | Différence RMS interne |

## Règles scientifiques
1. **Hypothèse obligatoire** : avant chaque modification, énoncer l'hypothèse
   mathématique (ex: "L'estimation conjointe piston+tilt par GLS réduira le
   biais résiduel sur les overlaps de s06").
2. **Pas de lissage massif** : interdiction de filtrage passe-bas global
   qui détruirait les mid-spatial frequencies.
3. **Pas de masquage opportuniste** : le `valid_mask` doit couvrir tout le
   support observé.
4. **Pas de modification des seeds** : les résultats doivent être reproductibles.
5. **Résumer l'échec** : si une tentative est rejetée, expliquer pourquoi en
   une phrase.
6. **Vectorisation** : privilégier NumPy/SciPy vectorisé + `scipy.sparse`.
   Pas de boucles Python sur les pixels.

## Vecteurs de recherche recommandés (par priorité)
1. Correction de piston par overlaps (déjà initié)
2. GLS simultané : piston + tip + tilt par moindres carrés
3. Régularisation de Tikhonov (explorer λ ∈ [1e-10, 1e-2])
4. Huber M-estimateur + IRLS pour robustesse aux outliers
5. Hybride CS/SC pour auto-calibration du biais de référence
6. Optimisation alternée pour retrace error
7. Pondération spatiale des overlaps (SNR-aware)

## Contrat du candidat
```python
class CandidateStitcher:
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface: ...
```

## Gouvernance qualitative
À RMS égal, préférer :
- Complexité cyclomatique plus faible
- Moins de lignes de code
- Usage de `scipy.sparse` plutôt que matrices denses
- Séparation claire des étapes (estimation → correction → assemblage)

## Glossaire

| Terme | Description |
|---|---|
| `footprint_iou` | Intersection‑over‑Union de la zone observée |
| `valid_pixel_recall` | Pixel recall valide |
| `valid_pixel_precision` | Pixel precision valide |
| `hf_retention` | Taux de rétention de haute fréquence |
| `runtime_sec` | Temps d’exécution par scénario |
| `mismatch_rms` | Différence RMS interne |
