# Project Intersection V1

Rete nuova per il progetto single-agent di controllo semaforico.

## Geometria

Per ciascuna direzione:

- tratto esterno: 2 corsie;
- tratto di avvicinamento: 3 corsie;
- corsia 0 (destra): svolta a destra oppure dritto;
- corsia 1 (centrale): dritto;
- corsia 2 (sinistra): solo svolta a sinistra protetta;
- uscita: 2 corsie.

Il semaforo `J0` è l'unico semaforo e sarà controllato da un unico agente RL.

## Fasi verdi principali

1. Nord-Sud: dritto e destra.
2. Nord-Sud: sinistre protette.
3. Est-Ovest: dritto e destra.
4. Est-Ovest: sinistre protette.

Le quattro fasi gialle sono presenti per il test SUMO. In `sumo-rl` l'ambiente potrà gestire automaticamente le transizioni.

## Creazione della rete

Dentro questa cartella:

```bash
chmod +x build_network.sh
./build_network.sh
```

In alternativa:

```bash
netconvert -c project_intersection.netccfg
```

## Primo test

Senza grafica:

```bash
sumo -c project_intersection.sumocfg
```

Con grafica:

```bash
sumo-gui -c project_intersection.sumocfg
```

## File

- `project_intersection.nod.xml`: nodi.
- `project_intersection.edg.xml`: strade e numero di corsie.
- `project_intersection.con.xml`: collegamenti e movimenti consentiti.
- `project_intersection.tll.xml`: programma semaforico iniziale.
- `project_intersection.netccfg`: configurazione di `netconvert`.
- `scenario_medium.rou.xml`: primo scenario di traffico.
- `project_intersection.sumocfg`: simulazione completa.

Pedoni, autobus, ambulanze e guidatori imperfetti verranno aggiunti nelle versioni successive, mantenendo la stessa rete e lo stesso agente.
