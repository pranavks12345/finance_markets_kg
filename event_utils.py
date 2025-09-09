from collections import defaultdict

def dedup_events(events):
    buckets = defaultdict(list)
    for e in events:
        key = (
            e["event_type"],
            (e["announcer"] or {}).get("normalized_id"),
            tuple(sorted(e.get("targets", [])))
        )
        buckets[key].append(e)

    merged = []
    for _, group in buckets.items():
        best = max(group, key=lambda x: x.get("confidence", 0))
        best["supporting_docs"] = sorted({g["source"]["doc_id"] for g in group if g.get("source")})
        best["confidence"] = max(g.get("confidence", 0) for g in group)
        merged.append(best)
    return merged