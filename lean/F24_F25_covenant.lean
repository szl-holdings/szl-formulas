/-
F24 and F25 are wedge-formula candidates. They are not locked formulas.

F24 is an induction on the worst-take fold: a list that contains `stale`
derives to `stale` or `unavailable`.

F25 follows from the definition of `reviewCycle`.
`f25_admitted_monotone` is the left inclusion of `++`.
`f25_no_unreviewed_entry` only shows that a newly present fact sits in the new
reviewed list, because that list appends the same filter as the new admitted
list. `f25_new_admit_was_reviewed` is the filter fact itself: the predicate
was `c ∈ reviewed` before the cycle. Those two are direct from that definition.

Status: CANDIDATE. Not entries in the locked formula reference.
No `sorry`. Each theorem in this file depends on `propext` and on no other axiom.
-/

namespace Puriq.Formula.Candidate

/- ========== F24: staleness propagation ========== -/

/-- The truth tiers. -/
inductive Tier where
  | fresh : Tier
  | stale : Tier
  | unavailable : Tier

/-- The composition rule: a derivation takes the worst tier of its inputs. -/
def Tier.worst : Tier → Tier → Tier
  | Tier.unavailable, _ => Tier.unavailable
  | _, Tier.unavailable => Tier.unavailable
  | Tier.stale, _ => Tier.stale
  | _, Tier.stale => Tier.stale
  | Tier.fresh, Tier.fresh => Tier.fresh

/-- A derivation over multiple inputs composes the worst-take rule. -/
def derive (ts : List Tier) : Tier := ts.foldr Tier.worst Tier.fresh

/-- Stale-or-worse: the property a derived result inherits from a stale input. -/
def StaleOrWorse : Tier → Prop
  | Tier.fresh => False
  | Tier.stale => True
  | Tier.unavailable => True

theorem worst_fresh_id (t : Tier) : Tier.worst Tier.fresh t = t := by
  cases t <;> rfl

/-- `worst stale t` is never `fresh`. The second argument still matters:
`unavailable` wins over `stale`. -/
theorem worst_stale_is_stale_or_worse (t : Tier) :
    StaleOrWorse (Tier.worst Tier.stale t) := by
  cases t with
  | fresh => exact True.intro
  | stale => exact True.intro
  | unavailable => exact True.intro

/-- If any input is stale, the derived result is stale or worse. -/
theorem f24_staleness_propagation (ts : List Tier) (h : Tier.stale ∈ ts) :
    StaleOrWorse (derive ts) := by
  induction ts with
  | nil => cases h
  | cons t ts ih =>
    cases t with
    | fresh =>
      have h' : Tier.stale ∈ ts := by
        cases List.mem_cons.mp h with
        | inl heq => exact Tier.noConfusion heq
        | inr h'' => exact h''
      show StaleOrWorse (Tier.worst Tier.fresh (derive ts))
      rw [worst_fresh_id]
      exact ih h'
    | stale =>
      exact worst_stale_is_stale_or_worse (derive ts)
    | unavailable => exact True.intro

/- ========== F25: learning monotonicity ========== -/

abbrev Fact := String

/-- A memory cycle: the admitted set and the reviewed set, with the review gate. -/
structure Memory where
  admitted : List Fact
  reviewed : List Fact
  gate : ∀ f ∈ admitted, f ∈ reviewed

/-- One review cycle: candidates already in `reviewed` are appended to both lists. -/
def Memory.reviewCycle (m : Memory) (candidates : List Fact) : Memory where
  admitted := m.admitted ++ (candidates.filter (fun c => c ∈ m.reviewed))
  reviewed := m.reviewed ++ (candidates.filter (fun c => c ∈ m.reviewed))
  gate := by
    intro f hf
    cases List.mem_append.mp hf with
    | inl h =>
      exact List.mem_append.mpr (Or.inl (m.gate f h))
    | inr h =>
      exact List.mem_append.mpr (Or.inr h)

/-- Definitional: the previous admitted list is a prefix of the new one. -/
theorem f25_admitted_monotone (m : Memory) (cands : List Fact) :
    ∀ f ∈ m.admitted, f ∈ (m.reviewCycle cands).admitted := by
  intro f hf
  exact List.mem_append.mpr (Or.inl hf)

/-- Definitional, and weaker than the informal wording. A fact that is new in
the admitted list is in the new reviewed list, because both fields append the
same filter. The filter predicate is not used. -/
theorem f25_no_unreviewed_entry (m : Memory) (cands : List Fact) :
    ∀ f ∈ (m.reviewCycle cands).admitted, f ∉ m.admitted → f ∈ (m.reviewCycle cands).reviewed := by
  intro f hf hout
  cases List.mem_append.mp hf with
  | inl h => exact absurd h hout
  | inr h => exact List.mem_append.mpr (Or.inr h)

/-- Direct from `List.filter`: a fact added by `reviewCycle` already belonged
to `m.reviewed`. -/
theorem f25_new_admit_was_reviewed (m : Memory) (cands : List Fact) :
    ∀ f ∈ (m.reviewCycle cands).admitted, f ∉ m.admitted → f ∈ m.reviewed := by
  intro f hf hout
  cases List.mem_append.mp hf with
  | inl h => exact absurd h hout
  | inr h =>
    exact of_decide_eq_true (List.mem_filter.mp h).2

end Puriq.Formula.Candidate
