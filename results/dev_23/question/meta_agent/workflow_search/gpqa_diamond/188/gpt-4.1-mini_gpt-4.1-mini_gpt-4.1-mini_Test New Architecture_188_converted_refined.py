async def forward_188(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Subtask 1: Analyze and classify each effective particle (Magnon, Skyrmion, Pion, Phonon) by explicitly identifying the specific symmetry spontaneously broken to produce it, "
        "the nature of the particle (Goldstone boson, topological excitation, vibrational mode, etc.), and the effective field theory or Hamiltonian context in which it arises. "
        "Explicitly state the vacuum manifold and broken symmetry for each particle, highlight the difference between spontaneous and explicit symmetry breaking, especially for phonons, to avoid conceptual errors. "
        "Address previous misclassifications of Skyrmions."
    )

    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }

    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Subtask 2: Verify the classifications from Subtask 1 by cross-checking whether each purported non-spontaneously broken symmetry (SSB) particle indeed does not require a spontaneously broken symmetry. "
        "Consult standard references or derive the vacuum manifold to confirm the origin of each particle and explicitly distinguish between spontaneous and explicit symmetry breaking. "
        "This verification step is designed to catch subtle misclassifications (e.g., phonons) before proceeding."
    )

    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }

    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Subtask 3: Based on the verified classifications from Stage 1, select and justify the final answer identifying which effective particle is not associated with a spontaneously-broken symmetry. "
        "Provide clear, well-reasoned explanations referencing the analysis and verification results, explicitly addressing subtle distinctions and ensuring no propagation of earlier conceptual errors."
    )

    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }

    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])

    return final_answer, logs
