# AGENTS.md — Autoresearch Optical Stitching

## Mission
Optimiser itérativement l'algorithme de stitching optique dans
`src/stitching/editable/candidate_current.py` en minimisant l'erreur RMS
agrégée sur la suite de scénarios.

**ÉTAT ACTUEL** : Une structure de solveur global (Piston uniquement) a été
pré-implémentée dans `candidate_current.py`.

**PROCHAINE ÉTAPE (PRIORITÉ)** : Étendre le solveur `_solve_global_alignment` pour
estimer également le **Tip, Tilt et Focus** (indices 1, 2, 3 du vecteur d'état
par sous-pupille). Voir `docs/stitching_implementation_guide.md` pour
le modèle mathématique complet.

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
7. **Prudence de Modélisation (Anti-Overfit)** : Ne pas corriger de termes (Focus, Astigmatisme...) s'ils ne sont pas nécessaires. Utiliser la **régularisation de Tikhonov** pour maintenir les paramètres d'alignement à zéro par défaut en l'absence de signal clair dans les overlaps.

## Vecteurs de recherche recommandés (par priorité)
1. **Solveur Global (Mandat)** : Résolution simultanée des vecteurs d'état
   $\mathbf{x} = [p_i, tx_i, ty_i, f_i]$ pour chaque sous-pupille $i$ en minimisant
   $\sum_{i,j} \iint_{O_{i,j}} (S_i(\mathbf{r}) - S_j(\mathbf{r}))^2 d\mathbf{r}$
   où $O_{i,j}$ est l'intersection des sous-pupilles $i$ et $j$.
2. **Robustesse (Huber/IRLS)** : Utiliser des poids itératifs pour ignorer les
   outliers et les bords de pupille dégradés dans le solveur.
3. **Pondération SNR** : Utiliser `detector_edge_roll_off` pour pondérer
   faiblement les bords dans la matrice normale.
4. **Auto-calibration** : Estimer un biais de référence statique commun à toutes
   les poses en même temps que les paramètres d'alignement.

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
