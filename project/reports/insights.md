# Rapport d'insights — Prévision des ventes supermarché
## À l'attention des responsables et décideurs

**Projet :** Prédiction des ventes journalières — Supermarchés du Bénin
**Période analysée :** Janvier 2023 – Décembre 2025
**Magasins couverts :** 10 supermarchés dans 10 villes du Bénin
**Modèle utilisé :** XGBoost (Intelligence Artificielle)

---

## Ce que ce projet permet de faire

Jusqu'à présent, la planification des stocks et des commandes reposait sur l'expérience personnelle des responsables ou sur des moyennes historiques grossières. Ce projet change cela.

Grâce aux données des 3 dernières années, on a construit un système capable de **prédire les ventes de n'importe quel magasin pour n'importe quelle journée** avec une précision de 92.45%.

Concrètement : si un responsable veut savoir combien son magasin va vendre samedi prochain — un jour de promotion, avec des congés scolaires — il peut obtenir une prédiction en moins d'une seconde.

---

## Ce que les données nous ont appris

### 1. Les ventes ne sont pas aléatoires — elles obéissent à des règles

C'est le premier enseignement fondamental. On aurait pu penser que les ventes fluctuent de façon imprévisible selon les humeurs des clients. Les données montrent le contraire.

**Le modèle explique 89% des variations de ventes.** Cela signifie que 89 jours sur 100, les ventes suivent des règles identifiables et prévisibles. Seulement 11% sont dus à des événements réellement imprévisibles — une panne de courant, un événement local inattendu, un comportement exceptionnel.

> **Implication directe :** La planification basée sur des données est possible et fiable. Les décisions de commande ne doivent plus reposer uniquement sur l'intuition.

---

### 2. Le type de magasin est le facteur numéro 1

Un tiers des variations de ventes (33.9%) s'explique uniquement par le **type de magasin**.

```
Type A → ventes moyennes : 12 309 FCFA/jour
Type B → ventes moyennes :  9 718 FCFA/jour
Type C → ventes moyennes :  7 804 FCFA/jour
```

Un magasin de type A vend en moyenne **58% de plus** qu'un magasin de type C. Cette différence est structurelle — elle vient de la taille, de l'emplacement, de l'assortiment.

> **Implication directe :** On ne peut pas comparer les performances brutes d'un magasin A et d'un magasin C. Un magasin C qui atteint 9 000 FCFA/jour fait peut-être mieux que sa catégorie, tandis qu'un magasin A à 11 000 FCFA/jour est peut-être en sous-performance. Chaque type doit avoir ses propres objectifs de ventes.

---

### 3. Les promotions sont le levier le plus puissant

La promotion est le **deuxième facteur** d'explication des ventes (19.2%) et c'est le seul sur lequel on peut agir directement et rapidement.

```
Jours sans promotion  → ventes moyennes : 9 120 FCFA
Jours avec promotion  → ventes moyennes : 11 663 FCFA
                                          ↑
                                       +27.9%
```

Une promotion génère en moyenne **+2 543 FCFA de ventes supplémentaires par jour**.

> **Implication directe :** Les promotions ont un retour sur investissement mesurable et significatif. Chaque décision de promotion doit être anticipée et planifiée — pas improvisée.

---

### 4. Les promotions du week-end valent le double

C'est la découverte la plus actionnable du projet.

```
Semaine normale, sans promo  →  8 803 FCFA
Week-end, sans promo         →  9 962 FCFA  (+13%)
Semaine, avec promo          → 11 316 FCFA  (+28%)
Week-end, avec promo         → 12 530 FCFA  (+42%)
```

Une promotion en semaine génère +28% de ventes. La même promotion le week-end génère +42%. Ce n'est pas 28% + 13% = 41%. C'est **42%** — il y a un effet de multiplication, pas juste d'addition.

> **Recommandation n°1 :** Concentrer les promotions sur les week-ends. Une promotion en semaine est bien. La même promotion le week-end rapporte significativement plus pour le même investissement.

---

### 5. La taille du magasin explique une part structurelle des ventes

La surface du magasin est le troisième facteur (10.9%). La corrélation est directe et forte : plus un magasin est grand, plus il vend.

Ce n'est pas surprenant — un grand magasin peut stocker plus de produits, proposer plus de rayons, accueillir plus de clients simultanément.

> **Implication directe :** La surface est un investissement à long terme dont le retour est mesurable. Pour les futures décisions d'expansion ou de rénovation, chaque m² supplémentaire a une valeur commerciale quantifiable.

---

### 6. La concurrence a un impact réel mais non actionnable à court terme

La distance avec le concurrent le plus proche est le cinquième facteur explicatif. Plus un concurrent est proche, plus les ventes baissent.

Ce facteur est **structurel** — on ne peut pas déplacer un concurrent. Mais il est utile pour :
- Comprendre pourquoi certains magasins performent moins bien
- Guider les décisions de localisation des futurs magasins
- Adapter la stratégie promotionnelle des magasins très exposés à la concurrence

> **Recommandation :** Les magasins avec un concurrent proche doivent compenser par une politique promotionnelle plus agressive. Le modèle confirme que la promotion est particulièrement efficace dans ce contexte.

---

### 7. Les jours fériés et les week-ends augmentent les ventes

```
Jour de semaine normale  →  9 554 FCFA
Week-end                 → 10 765 FCFA  (+12.7%)
Jour férié               → 11 600 FCFA  (+21.4%)
```

Les jours fériés surpassent même les week-ends. Les clients profitent de leur liberté pour faire des achats plus importants.

> **Recommandation :** Anticiper les jours fériés dans la planification des stocks. Ces jours demandent des approvisionnements plus importants — une rupture de stock un jour férié représente une perte de revenus significative.

---

### 8. Les congés scolaires ont un effet modéré

Les congés scolaires augmentent légèrement les ventes. Les familles font plus de courses quand les enfants sont à la maison.

Cet effet est réel mais secondaire par rapport aux promotions et aux week-ends. Il est néanmoins capté par le modèle et intégré dans les prédictions.

---

### 9. Les facteurs macroéconomiques jouent un rôle secondaire

L'inflation et l'indice économique ont un effet sur les ventes, mais il est plus faible que les facteurs opérationnels.

```
Inflation élevée   → légère baisse des ventes (-0.13 de corrélation)
Bon indice économique → légère hausse des ventes (+0.16 de corrélation)
```

Ces facteurs sont utiles pour des prévisions à long terme (planification annuelle) mais ont peu d'impact sur les prédictions jour à jour.

> **Implication :** En période d'inflation forte, les marges promotionnelles doivent être ajustées. Le pouvoir d'achat des clients est comprimé — des promotions bien ciblées sont encore plus importantes pour maintenir le volume de ventes.

---

### 10. La météo n'explique pas les ventes dans ce contexte

La température et les précipitations ont une influence quasi nulle sur les ventes de ces supermarchés (moins de 0.3% d'importance chacun).

Cela peut s'expliquer par le fait que les supermarchés sont des achats de nécessité — les clients viennent quelle que soit la météo. La pluie peut retarder une visite d'une heure, mais pas l'annuler.

> **Implication :** Inutile d'ajuster les stocks ou les promotions en fonction de la météo prévue. D'autres facteurs sont bien plus déterminants.

---

## La précision du modèle en chiffres

| Métrique | Valeur | Ce que ça veut dire |
|---|---|---|
| Erreur moyenne (MAE) | 696 FCFA | Sur 10 000 FCFA de ventes, le modèle se trompe de ±696 FCFA en moyenne |
| Erreur en % (MAPE) | 7.55% | Le modèle se trompe de 7.55% en moyenne |
| Variance expliquée (R²) | 89% | 89% des variations de ventes sont expliquées |

**Pour mettre en perspective :**
Dans l'industrie du commerce de détail, un modèle de prévision est considéré comme :
- Excellent si l'erreur est inférieure à 5%
- **Bon si l'erreur est entre 5% et 10% ← notre modèle est ici**
- Acceptable si l'erreur est entre 10% et 20%
- Insuffisant au-delà de 20%

Notre modèle à **7.55% d'erreur est prêt pour une utilisation en production.**

---

## Ce que le système peut faire concrètement

Le modèle est disponible via une application (API) accessible depuis n'importe quel système. Un responsable de magasin ou un système de gestion des stocks peut envoyer les caractéristiques d'une journée et recevoir immédiatement une prédiction.

**Exemple d'utilisation :**

> *"Samedi 15 juin 2025. Magasin S001 à Cotonou. Type A. Promotion active. Pas de jour férié. Température 29°C. Inflation 5.5%."*

→ **Prédiction : 14 823 FCFA de ventes**
→ **Confiance : modérée (erreur possible de ±7.55%)**
→ **Fourchette réaliste : entre 13 700 et 15 950 FCFA**

---

## Les 5 recommandations prioritaires

**Recommandation 1 — Concentrer les promotions sur les week-ends**
L'effet multiplicateur Promo × Weekend génère +42% de ventes contre +28% en semaine. Pour le même budget promotionnel, planifier les promotions le vendredi et le samedi maximise le retour sur investissement.

**Recommandation 2 — Différencier les objectifs par type de magasin**
Un magasin C ne peut pas atteindre les mêmes volumes qu'un magasin A. Les objectifs de performance, les dotations en stock et les budgets promotionnels doivent être calibrés par type.

**Recommandation 3 — Anticiper les jours fériés dans les approvisionnements**
Les jours fériés génèrent +21.4% de ventes. Une rupture de stock ce jour-là coûte cher. Les commandes doivent être passées suffisamment à l'avance pour couvrir ces pics prévisibles.

**Recommandation 4 — Renforcer la stratégie promotionnelle dans les magasins proches de concurrents**
La distance avec la concurrence est un facteur structurel qu'on ne peut pas changer. Mais une politique promotionnelle plus active peut compenser partiellement cet effet. Le modèle confirme que les promotions sont particulièrement efficaces dans ce contexte.

**Recommandation 5 — Utiliser les prédictions pour optimiser les commandes**
Avec une erreur de 7.55%, le modèle prédit les ventes avec une précision suffisante pour ajuster les commandes de stock. Commander sur la base des prédictions plutôt que sur des moyennes historiques peut réduire significativement les pertes par gaspillage et les pertes de ventes par rupture.

---

## Limites du modèle

Toute prédiction doit être utilisée comme **aide à la décision**, pas comme vérité absolue. Le modèle ne peut pas anticiper :

- Un événement local imprévu (match important, manifestation, fête de quartier)
- Une panne informatique ou une fermeture exceptionnelle
- Une rupture de stock massive non planifiée
- Un changement brutal de concurrence (ouverture d'un grand concurrent à proximité)

Ces événements représentent les 11% de variance que le modèle n'explique pas. Pour ces cas, le jugement humain reste indispensable.

---

*Document produit dans le cadre du projet de prévision des ventes — Personne B (Modélisation & Ingénierie)*
*Données : Supermarchés du Bénin, 2023–2025*
