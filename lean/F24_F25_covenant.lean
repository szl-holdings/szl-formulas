/-
F24: Staleness propagation — a stale source cannot become fresh by derivation.
F25: Learning monotonicity — review-gated memory only grows; the unreviewed
     can never enter.

Carriers for claims F24/F25 in contracts/wedge-formula-candidates.v1.json.
Axioms: none beyond Lean core. Zero sorry.
-/

namespace Puriq.Formula.Proved

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

/-- THE WEDGE THEOREM F24: if any input is stale, the derived result is
stale or worse. Staleness cannot be laundered by derivation. -/
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
    | stale => trivial
    | unavailable => trivial

/- ========== F25: learning monotonicity ========== -/

abbrev Fact := String

/-- A memory cycle: the admitted set and the reviewed set, with the review gate. -/
structure Memory where
  admitted : List Fact
  reviewed : List Fact
  gate : ∀ f ∈ admitted, f ∈ reviewed

/-- One review cycle: candidates that pass review are admitted; nothing else enters. -/
def Memory.reviewCycle (m : Memory) (candidates : List Fact) : Memory where
  admitted := m.admitted ++ (candidates.filter (fun c => c ∈ m.reviewed))
  reviewed := m.reviewed ++ (candidates.filter (fun c => c ∈ m.reviewed))
  gate := by
    intro f hf
    cases List.mem_append.mp hf with
    | inl h => exact m.gate f h
    | inr h => exact List.mem_append.mpr (Or.inr h)

/-- THE WEDGE THEOREM F25a: the admitted set only grows across cycles. -/
theorem f25_admitted_monotone (m : Memory) (cands : List Fact) :
    ∀ f ∈ m.admitted, f ∈ (m.reviewCycle cands).admitted := by
  intro f hf
  exact List.mem_append.mpr (Or.inl hf)

/-- THE WEDGE THEOREM F25b: nothing unreviewed ever enters the admitted set. -/
theorem f25_no_unreviewed_entry (m : Memory) (cands : List Fact) :
    ∀ f ∈ (m.reviewCycle cands).admitted, f ∉ m.admitted → f ∈ (m.reviewCycle cands).reviewed := by
  intro f hf hout
  cases List.mem_append.mp hf with
  | inl h => exact absurd h hout
  | inr h => exact List.mem_append.mpr (Or.inr h)

end Puriq.Formula.Proved
