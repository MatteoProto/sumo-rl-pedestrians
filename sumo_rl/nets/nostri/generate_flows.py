#!/usr/bin/env python3
"""
generate_flows.py
==================
Genera automaticamente un file .rou.xml (veicoli + pedoni) per QUALSIASI
rete SUMO a partire dal solo file .net.xml, riconoscendo da solo i bracci
di ingresso/uscita e producendo un mix di traffico con 5 tipi di veicolo
e pedoni con comportamento SUMO di default (tipo built-in "pedestrian").

Richiede: sumolib (pip install sumolib)
I vType veicolari devono essere definiti in vtypes.xml (vedi file allegato),
incluso nel rou.xml con <include href="vtypes.xml"/>.

USO BASE
--------
    python3 generate_flows.py -n rete.net.xml -o rotte.rou.xml

OPZIONI PRINCIPALI
-------------------
    -n, --net               file .net.xml di input (obbligatorio)
    -o, --output            file .rou.xml di output (default: <net>_flows.rou.xml)
    --begin / --end         intervallo simulazione in secondi (default 0-3600)
    --through-volume        veicoli/h per movimento "dritto" (default 260)
    --turn-volume           veicoli/h per movimento "svolta 90°" (default 90)
    --diag-volume           veicoli/h per movimento "diagonale" (default 0)
    --ped-through-period    periodo medio (s) tra pedoni asse dritto (default 120)
    --ped-turn-period       periodo medio (s) per pedoni in svolta (default 200)
    --ped-diag-period       periodo medio (s) per pedoni diagonale (default 400)
    --no-pedestrians        non generare flussi pedonali
    --no-vehicles           non generare flussi veicolari
    --vtypes-file           percorso del file vtypes (default "vtypes.xml")

LOGICA DI CLASSIFICAZIONE DEI MOVIMENTI
----------------------------------------
Per ogni coppia (arco ingresso, arco uscita):
  - esclude inversioni a U
  - calcola angolo tra heading in uscita e in ingresso
  - "through"  = angolo vicino a 0°/360° (prosecuzione dritta)
  - "turn"     = angolo vicino a 90°/270° (svolta singola)
  - "diagonal" = angoli intermedi (incroci con più di 4 bracci)
"""

import argparse
import math
import sys
from xml.sax.saxutils import quoteattr

import sumolib


# ---------------------------------------------------------------------------
# Mix percentuali veicolari per classe di movimento (somma = 1.0)
# ---------------------------------------------------------------------------
VEHICLE_MIX = {
    "through":  [("auto_normale", 0.566), ("auto_aggressivo", 0.189),
                 ("auto_prudente", 0.113), ("camion", 0.075), ("moto", 0.057)],
    "turn":     [("auto_normale", 0.682), ("auto_aggressivo", 0.227), ("moto", 0.091)],
    "diagonal": [("auto_normale", 0.77),  ("auto_aggressivo", 0.23)],
}


def classify_angle(angle_deg):
    """
    Classifica angolo di svolta (0-360): 0°/360° = dritto, 90°/270° = svolta.
    """
    d0  = min(angle_deg, 360 - angle_deg)
    d90 = min(abs(((angle_deg - 90)  + 180) % 360 - 180),
              abs(((angle_deg - 270) + 180) % 360 - 180))
    if d0  <= 30: return "through"
    if d90 <= 30: return "turn"
    return "diagonal"


def edge_heading(edge, at_start):
    shape = edge.getShape()
    if at_start:
        (x1, y1), (x2, y2) = shape[0], shape[1] if len(shape) > 1 else shape[0]
    else:
        (x1, y1), (x2, y2) = shape[-2] if len(shape) > 1 else shape[-1], shape[-1]
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360


def get_boundary_edges(net):
    entry, exit_ = [], []
    for edge in net.getEdges():
        if edge.getFunction() != "":
            continue
        if edge.getFromNode().getType() == "dead_end":
            entry.append(edge)
        if edge.getToNode().getType() == "dead_end":
            exit_.append(edge)
    if not entry or not exit_:
        sys.stderr.write(
            "ATTENZIONE: nessun nodo 'dead_end' trovato. "
            "Uso tutti gli archi normali come bordo.\n")
        entry = exit_ = [e for e in net.getEdges() if e.getFunction() == ""]
    return entry, exit_


def has_pedestrian_lane(edge):
    return any(lane.allows("pedestrian") for lane in edge.getLanes())


def build_routes(net, entry_edges, exit_edges):
    movements = []
    for e_in in entry_edges:
        for e_out in exit_edges:
            if e_in.getID() == e_out.getID():
                continue
            if e_in.getFromNode().getID() == e_out.getToNode().getID():
                continue
            path, _ = net.getShortestPath(e_in, e_out, vClass="passenger")
            if not path:
                continue
            h_in  = edge_heading(e_in,  at_start=False)
            h_out = edge_heading(e_out, at_start=True)
            angle = (h_out - h_in) % 360
            mclass = classify_angle(angle)
            movements.append({
                "from":     e_in,
                "to":       e_out,
                "class":    mclass,
                "edges":    " ".join(e.getID() for e in path),
                "ped_ok":   has_pedestrian_lane(e_in) and has_pedestrian_lane(e_out),
            })
    return movements


def write_rou_xml(movements, args):
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!--')
    lines.append(f'    Generato da generate_flows.py per la rete: {args.net}')
    lines.append('    Pedoni: tipo built-in SUMO "pedestrian" (default).')
    lines.append('-->')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    lines.append('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append(f'    <include href={quoteattr(args.vtypes_file)}/>')
    lines.append("")

    veh_volume = {"through": args.through_volume, "turn": args.turn_volume,
                  "diagonal": args.diag_volume}
    ped_period  = {"through": args.ped_through_period, "turn": args.ped_turn_period,
                   "diagonal": args.ped_diag_period}

    # ---- rotte ----
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
                    f'begin="{args.begin}" end="{args.end}" probability="{round(vph/3600, 6)}" '
                    f'departLane="best" departSpeed="desired"/>')
        lines.append("")

    # ---- flussi pedonali (tipo default SUMO, un solo flusso per movimento) ----
    if not args.no_pedestrians:
        lines.append("    <!-- ===================== FLUSSI PEDONALI ===================== -->")
        for i, m in enumerate(movements):
            if not m["ped_ok"]:
                continue
            period = ped_period[m["class"]]
            if period <= 0:
                continue
            pid = f"pf{i:03d}"
            # exp(X) in SUMO = rate λ=X (veicoli/pedoni al secondo).
            # Per avere media=period secondi, il rate è 1/period.
            lines.append(
                f'    <personFlow id="{pid}" '
                f'begin="{args.begin}" end="{args.end}" period="exp({round(1.0/period, 6)})">')
            lines.append(
                f'        <walk from="{m["from"].getID()}" to="{m["to"].getID()}"/>')
            lines.append('    </personFlow>')
        lines.append("")

    lines.append('</routes>')
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--net",      required=True)
    p.add_argument("-o", "--output",   default=None)
    p.add_argument("--vtypes-file",    default="vtypes.xml")
    p.add_argument("--begin",          type=int,   default=0)
    p.add_argument("--end",            type=int,   default=3600)
    p.add_argument("--through-volume", type=float, default=260)
    p.add_argument("--turn-volume",    type=float, default=90)
    p.add_argument("--diag-volume",    type=float, default=0)
    p.add_argument("--ped-through-period", type=float, default=30,
                   help="periodo medio (s) tra pedoni asse dritto (default 30 → ~120 ped/h)")
    p.add_argument("--ped-turn-period",    type=float, default=200,
                   help="periodo medio (s) pedoni in svolta (default 200 → ~18 ped/h)")
    p.add_argument("--ped-diag-period",    type=float, default=400,
                   help="periodo medio (s) pedoni diagonale (default 400 → ~9 ped/h)")
    p.add_argument("--no-pedestrians", action="store_true")
    p.add_argument("--no-vehicles",    action="store_true")
    args = p.parse_args()

    if not args.output:
        args.output = args.net.replace(".net.xml", "") + "_flows.rou.xml"

    net = sumolib.net.readNet(args.net, withInternal=False)
    entry_edges, exit_edges = get_boundary_edges(net)
    print(f"Bracci di ingresso: {[e.getID() for e in entry_edges]}")
    print(f"Bracci di uscita:   {[e.getID() for e in exit_edges]}")

    movements = build_routes(net, entry_edges, exit_edges)
    print(f"Movimenti O-D: {len(movements)}")
    for m in movements:
        print(f"  {m['from'].getID():>6} -> {m['to'].getID():<6} "
              f"[{m['class']:8}] pedoni={'si' if m['ped_ok'] else 'no'}")

    xml = write_rou_xml(movements, args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"\nFile scritto: {args.output}")


if __name__ == "__main__":
    main()