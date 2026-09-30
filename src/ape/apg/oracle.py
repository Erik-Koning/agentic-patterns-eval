"""Oracle APG graphs built directly from the world spec (arm S5o: an upper bound only).

Authoring rules, dictated by how `compose` works (READINESS gaps G1-G3):
- knowledge lives on routable leaves; category nodes are thin, non-routable descriptors
  (compose never includes a target's descendants);
- relations (exception <-> policy, procedure -> tool spec) are leaf-level `bring` edges;
- tool allowlists sit on leaves only (the allowlist is intersected along the primary path).
The harness system prompt carries persona and task for every arm, so the root has no slots.
"""

import copy

from apg_core import validate_graph

from ..worlds.spec import World


def _doc(world: World, nodes: list[dict], profile: str, budget: int) -> dict:
    return {
        "schemaVersion": "1.0",
        "graphId": f"{world.id}.oracle",
        "version": "1",
        "profile": profile,
        "defaults": {
            "routing": {"minConfidence": 0.55, "allowMulti": True, "shortlistK": 12},
            "budget": {"maxPromptTokens": budget},
        },
        "nodes": nodes,
        "edges": [],
    }


def _root(world: World) -> dict:
    return {"id": "root", "parentId": None, "routable": False, "title": f"Company knowledge ({world.family})"}


def _category(cid: str, title: str, description: str) -> dict:
    return {"id": cid, "parentId": "root", "type": "category", "routable": False, "title": title, "description": description}


def _leaf(nid: str, parent: str, title: str, description: str, knowledge: str, facts: list[str], **extra) -> dict:
    node = {
        "id": nid,
        "parentId": parent,
        "type": "category",
        "routable": True,
        "title": title,
        "description": description,
        "prompt": {"slots": {"knowledge": knowledge}},
        "props": {"factIds": facts},
    }
    node.update(extra)
    return node


def build_oracle(world: World, budget_tokens: int) -> dict:
    nodes = [_root(world)]
    if world.family == "F7":
        exc_by_policy: dict[str, list] = {}
        for x in world.exceptions:
            exc_by_policy.setdefault(x.policy_id, []).append(x)
        for dom in sorted({p.domain for p in world.policies}):
            nodes.append(_category(f"dom-{dom}", f"{dom.replace('_', ' ').title()} policies", f"Handling rules for {dom.replace('_', ' ')} requests."))
        for p in world.policies:
            excs = exc_by_policy.get(p.id, [])
            nodes.append(
                _leaf(
                    p.id,
                    f"dom-{p.domain}",
                    f"Policy {p.id}: {p.domain.replace('_', ' ')} requests, {p.region} region, ${p.lo:,} to under ${p.hi:,}",
                    f"{p.domain.replace('_', ' ')} request in the {p.region} region with an amount from ${p.lo:,} to under ${p.hi:,}",
                    world.facts[p.fact_id].text,
                    [p.fact_id],
                    **({"bring": [x.id for x in excs]} if excs else {}),
                )
            )
            for x in excs:
                nodes.append(
                    _leaf(
                        x.id,
                        f"dom-{p.domain}",
                        f"Exception {x.id} to Policy {p.id} for {x.tier} customers",
                        f"{p.domain.replace('_', ' ')} request, {p.region} region, {x.tier} status, amends policy {p.id}",
                        world.facts[x.fact_id].text,
                        [x.fact_id],
                        bring=[p.id],
                    )
                )
        profile = "L1"
    elif world.family == "F3":
        tools = {t.name: t for t in world.tools}
        for dom in sorted({t.domain for t in world.tools}):
            nodes.append(_category(f"dom-{dom}", f"{dom.replace('_', ' ').title()} operations", f"Procedures and tools for {dom.replace('_', ' ')} tickets."))
        for t in world.tools:
            leaf = _leaf(f"tool-{t.name}", f"dom-{t.domain}", f"Tool {t.name}", t.description, world.facts[t.fact_id].text, [t.fact_id])
            leaf["routable"] = False  # tool specs arrive only via a procedure's bring
            nodes.append(leaf)
        for proc in world.procedures:
            step_tools = [s["tool"] for s in proc.steps]
            dom, region = proc.situation.split("/")
            nodes.append(
                _leaf(
                    proc.id,
                    f"dom-{proc.domain}",
                    f"Procedure {proc.id}: {dom.replace('_', ' ')} tickets for orders shipping to {region}",
                    f"{dom.replace('_', ' ')} ticket, order ships to the {region} region",
                    world.facts[proc.fact_id].text,
                    [proc.fact_id],
                    bring=[f"tool-{n}" for n in step_tools if n in tools],
                    toolAllowlist=step_tools,
                )
            )
        profile = "L3"
    elif world.family == "F5":
        for team in sorted({e.subject for e in world.events}):
            evs = sorted((e for e in world.events if e.subject == team), key=lambda e: (e.date, e.id))
            managers = sorted({e.value for e in evs if e.relation == "manager"})
            nodes.append(
                _leaf(
                    f"team-{team.lower()}",
                    "root",
                    f"{team} team history",
                    f"Managers and office locations of the {team} team over time",
                    "\n".join(world.facts[e.fact_id].text for e in evs),
                    [e.fact_id for e in evs],
                    aliases=managers,
                )
            )
        profile = "L1"
    else:
        raise ValueError(world.family)
    doc = _doc(world, nodes, profile, budget_tokens)
    check = validate_graph(doc)
    if not check["valid"]:
        raise ValueError(f"oracle graph invalid: {check['errors'][:3]}")
    return doc


def strip_embeddings(doc: dict) -> dict:
    d = copy.deepcopy(doc)
    for n in d["nodes"]:
        n.pop("embedding", None)
    return d
