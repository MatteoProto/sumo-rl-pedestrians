#!/usr/bin/env python3
"""
generate_flows.py
==================
Genera automaticamente un file .rou.xml (veicoli + pedoni) per QUALSIASI
rete SUMO a partire dal solo file .net.xml, riconoscendo da solo i bracci
di ingresso/uscita e producendo un mix di traffico analogo a quello
impostato manualmente per l'incrocio J3 (5 tipi di auto, 3 tipi di
pedone, volumi differenziati per dritto/svolta/diagonale).

Richiede: sumolib (pip install sumolib)
I vType usati devono essere definiti in vtypes.xml (vedi file allegato),
incluso a parte nel comando sumo (-a vtypes.xml).

USO BASE
--------
    python3 generate_flows.py -n rete.net.xml -o rotte.rou.xml

OPZIONI PRINCIPALI
-------------------
    -n, --net           file .net.xml di input (obbligatorio)
    -o, --output         file .rou.xml di output (default: <net>_flows.rou.xml)
    --begin / --end       intervallo di simulazione in secondi (default 0-3600)
    --through-volume      veicoli/h totali per movimento "dritto" (default 260)
    --turn-volume         veicoli/h totali per movimento "svolta 90°" (default 90)
    --diag-volume         veicoli/h totali per movimento "diagonale/raro" (default 0,
                           cioè le svolte ad angolo molto stretto non generano
                           traffico veicolare a meno di impostare un valore > 0)
    --ped-through-period  periodo medio (s) tra un pedone e l'altro sull'asse
                           principale (default 25 → ~144 ped/h)
    --ped-turn-period     periodo medio (s) per i pedoni in diagonale agli angoli
                           (default 45)
    --ped-diag-period     periodo medio (s) per i pedoni che attraversano l'intero
                           incrocio (default 85)
    --no-pedestrians      non generare flussi pedonali (solo veicoli)
    --no-vehicles         non generare flussi veicolari (solo pedoni)
    --seed                seed per la randomizzazione dei mix percentuali
    --vtypes-file          nome/percorso del file vtypes da referenziare con <include>
                           (default "vtypes.xml")

LOGICA DI CLASSIFICAZIONE DEI MOVIMENTI
----------------------------------------
Per ogni coppia (arco di ingresso, arco di uscita) sulla rete, lo script:
  1. esclude le inversioni a U (stesso "capolinea" di origine e destinazione)
  2. calcola l'angolo tra la direzione di arrivo e quella di partenza
  3. classifica il movimento come:
       - "through"  se l'angolo è vicino a 180° (prosecuzione dritta)
       - "turn"     se l'angolo è vicino a 90°/270° (svolta singola, es. 4 bracci)
       - "diagonal" altrimenti (incroci con più di 4 bracci, angoli intermedi)
  4. instrada con il percorso più breve (sumolib) tra i due archi
  5. assegna volumi/percentuali per tipo di veicolo/pedone secondo la classe

Questo rende lo script indipendente dal numero di bracci e dai nomi degli
archi: funziona allo stesso modo su un incrocio a 3, 4 o 6 bracci, una
rotonda, o una rete con più intersezioni (in tal caso i bracci "di bordo"
sono quelli che terminano in un dead end o comunque non hanno ulteriori
collegamenti in una direzione).
"""

import argparse
import math
import random
import sys
from xml.sax.saxutils import quoteattr

import sumolib


# ---------------------------------------------------------------------------
# Mix percentuali per classe di movimento (somma = 1.0 per ciascuna classe)
# Replica le proporzioni usate nel file di esempio per l'incrocio a 4 bracci.
# ---------------------------------------------------------------------------
VEHICLE_MIX = {
    "through": [("auto_normale", 0.566), ("auto_aggressivo", 0.189),
                ("auto_prudente", 0.113), ("camion", 0.075), ("moto", 0.057)],
    "turn":    [("auto_normale", 0.682), ("auto_aggressivo", 0.227), ("moto", 0.091)],
    "diagonal": [("auto_normale", 0.77), ("auto_aggressivo", 0.23)],
}

# Per i pedoni: quali sottotipi includere per classe e relativo fattore di
# periodo rispetto al periodo "base" passato da CLI per quella classe
# (fattore 1.0 = stesso periodo medio del normale, >1 = più raro).
PEDESTRIAN_MIX = {
    "through":  [("pedone_normale", 1.0), ("pedone_veloce", 2.3), ("pedone_lento", 3.5)],
    "turn":     [("pedone_normale", 1.0), ("pedone_lento", 2.6)],
    "diagonal": [("pedone_normale", 1.0)],
}


def classify_angle(angle_deg):
    """
    Classifica un angolo di svolta (differenza tra heading in uscita e in
    ingresso, 0-360). 0°/360° = prosecuzione dritta (stessa direzione),
    90°/270° = svolta singola, valori intermedi = diagonale/svolta stretta.
    """
    d0 = min(angle_deg, 360 - angle_deg)               # distanza da 0°/360°
    d90 = min(abs(((angle_deg - 90) + 180) % 360 - 180),
              abs(((angle_deg - 270) + 180) % 360 - 180))  # distanza da 90° o 270°
    if d0 <= 30:
        return "through"
    if d90 <= 30:
        return "turn"
    return "diagonal"


def edge_heading(edge, at_start):
    """Heading approssimativo (gradi) dell'arco all'inizio o alla fine."""
    shape = edge.getShape()
    if at_start:
        (x1, y1), (x2, y2) = shape[0], shape[1] if len(shape) > 1 else shape[0]
    else:
        (x1, y1), (x2, y2) = shape[-2] if len(shape) > 1 else shape[-1], shape[-1]
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360


def get_boundary_edges(net):
    """
    Ritorna (entry_edges, exit_edges): liste di edge "di bordo" della rete,
    cioè quelli collegati a un junction senza altre uscite/entrate
    (dead_end) o comunque privi di archi opposti utilizzabili come
    prosecuzione interna. Funziona sia su singoli incroci sia su reti
    multi-giunzione, purché i bracci esterni terminino in nodi dead_end
    (caso tipico delle reti generate con netedit/netconvert per simulazioni
    isolate, come nel net.xml fornito).
    """
    entry, exit_ = [], []
    for edge in net.getEdges():
        if edge.getFunction() != "":  # salta archi interni/crossing/walkingarea
            continue
        from_node, to_node = edge.getFromNode(), edge.getToNode()
        if from_node.getType() == "dead_end":
            entry.append(edge)
        if to_node.getType() == "dead_end":
            exit_.append(edge)
    if not entry or not exit_:
        sys.stderr.write(
            "ATTENZIONE: nessun nodo 'dead_end' trovato. Uso come bordo tutti "
            "gli archi normali (potrebbe generare troppe combinazioni su reti "
            "interne complesse).\n")
        entry = exit_ = [e for e in net.getEdges() if e.getFunction() == ""]
    return entry, exit_


def has_pedestrian_lane(edge):
    return any(lane.allows("pedestrian") for lane in edge.getLanes())


def build_routes(net, entry_edges, exit_edges):
    """Per ogni coppia valida (entry, exit) calcola route + classe movimento."""
    movements = []
    for e_in in entry_edges:
        for e_out in exit_edges:
            if e_in.getID() == e_out.getID():
                continue
            # esclude inversioni a U: stesso nodo di bordo origine/destinazione
            if e_in.getFromNode().getID() == e_out.getToNode().getID():
                continue
            path, cost = net.getShortestPath(e_in, e_out, vClass="passenger")
            if not path:
                continue
            h_in = edge_heading(e_in, at_start=False)
            h_out = edge_heading(e_out, at_start=True)
            angle = (h_out - h_in) % 360
            mclass = classify_angle(angle)
            movements.append({
                "from": e_in, "to": e_out, "class": mclass,
                "edges": " ".join(e.getID() for e in path),
                "ped_ok": has_pedestrian_lane(e_in) and has_pedestrian_lane(e_out),
            })
    return movements


def write_rou_xml(movements, args):
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!--')
    lines.append(f'    Generato automaticamente da generate_flows.py per la rete: {args.net}')
    lines.append('    Mix veicolare/pedonale parametrico, indipendente dalla geometria della rete.')
    lines.append('-->')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    lines.append('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append(f'    <include href={quoteattr(args.vtypes_file)}/>')
    lines.append("")

    veh_volume = {"through": args.through_volume, "turn": args.turn_volume,
                  "diagonal": args.diag_volume}
    ped_period = {"through": args.ped_through_period, "turn": args.ped_turn_period,
                  "diagonal": args.ped_diag_period}

    # ---- rotte + flussi veicolari ----
    if not args.no_vehicles:
        lines.append("    <!-- ===================== ROTTE ===================== -->")
        for i, m in enumerate(movements):
            rid = f"r{i:03d}_{m['from'].getID()}_{m['to'].getID()}"
            m["route_id"] = rid
            lines.append(f'    <route id="{rid}" edges="{m["edges"]}"/>')
        lines.append("")
        lines.append("    <!-- ===================== FLUSSI VEICOLARI ===================== -->")
        for i, m in enumerate(movements):
            total = veh_volume[m["class"]]
            if total <= 0:
                continue
            for vtype, frac in VEHICLE_MIX[m["class"]]:
                vph = round(total * frac, 2)
                if vph <= 0:
                    continue
                fid = f"f{i:03d}_{vtype}"
                lines.append(
                    f'    <flow id="{fid}" route="{m["route_id"]}" type="{vtype}" '
                    f'begin="{args.begin}" end="{args.end}" vehsPerHour="{vph}" '
                    f'departLane="best" departSpeed="desired"/>')
        lines.append("")

    # ---- flussi pedonali ----
    if not args.no_pedestrians:
        lines.append("    <!-- ===================== FLUSSI PEDONALI ===================== -->")
        for i, m in enumerate(movements):
            if not m["ped_ok"]:
                continue
            base_period = ped_period[m["class"]]
            if base_period <= 0:
                continue
            for ptype, factor in PEDESTRIAN_MIX[m["class"]]:
                period = round(base_period * factor, 1)
                pid = f"pf{i:03d}_{ptype}"
                lines.append(
                    f'    <personFlow id="{pid}" type="{ptype}" '
                    f'begin="{args.begin}" end="{args.end}" period="exp({period})">')
                lines.append(
                    f'        <walk from="{m["from"].getID()}" to="{m["to"].getID()}"/>')
                lines.append('    </personFlow>')
        lines.append("")

    lines.append('</routes>')
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--net", required=True, help="file .net.xml di input")
    p.add_argument("-o", "--output", default=None, help="file .rou.xml di output")
    p.add_argument("--vtypes-file", default="vtypes.xml",
                   help="percorso del file vtypes da referenziare con <include>")
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=3600)
    p.add_argument("--through-volume", type=float, default=260)
    p.add_argument("--turn-volume", type=float, default=90)
    p.add_argument("--diag-volume", type=float, default=0)
    p.add_argument("--ped-through-period", type=float, default=25)
    p.add_argument("--ped-turn-period", type=float, default=45)
    p.add_argument("--ped-diag-period", type=float, default=85)
    p.add_argument("--no-pedestrians", action="store_true")
    p.add_argument("--no-vehicles", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.output:
        args.output = args.net.replace(".net.xml", "") + "_flows.rou.xml"

    net = sumolib.net.readNet(args.net, withInternal=False)
    entry_edges, exit_edges = get_boundary_edges(net)
    print(f"Bracci di ingresso trovati: {[e.getID() for e in entry_edges]}")
    print(f"Bracci di uscita trovati:   {[e.getID() for e in exit_edges]}")

    movements = build_routes(net, entry_edges, exit_edges)
    print(f"Movimenti O-D generati: {len(movements)}")
    for m in movements:
        print(f"  {m['from'].getID():>6} -> {m['to'].getID():<6} "
              f"[{m['class']:8}] pedoni={'si' if m['ped_ok'] else 'no'}")

    xml = write_rou_xml(movements, args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"\nFile scritto: {args.output}")


if __name__ == "__main__":
    main()
