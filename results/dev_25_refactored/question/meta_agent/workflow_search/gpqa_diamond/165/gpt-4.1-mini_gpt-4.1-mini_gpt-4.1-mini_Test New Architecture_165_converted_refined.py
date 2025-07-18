async def forward_165(self, taskInfo):
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 1: Extract and transform the given Lagrangian, field representations, and vacuum expectation values "
        "into a structured theoretical framework suitable for analyzing radiative corrections. Ensure precise identification "
        "of all fields, their gauge representations, and symmetry breaking scales. This step sets the foundation for subsequent analysis and must avoid assumptions about couplings or contributions."
    )
    cot_sc_desc_0_1 = {
        'instruction': cot_sc_instruction_0_1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent theoretical framework representation.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc_0_1,
        n_repeat=self.max_sc
    )
    logs.append(log_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the structured theoretical framework from Sub-task 1, derive the interaction vertices of the pseudo-Goldstone boson H2 "
        "with each particle potentially contributing to radiative corrections (W, Z, h1, H+, H0, A0, t, Ni). Explicitly verify which particles have nonzero couplings to H2, "
        "clarifying the physical origin of each interaction."
    )
    cot_sc_desc_0_2 = {
        'instruction': cot_sc_instruction_0_2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent set of interaction vertices.",
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Using the derived interaction vertices from stage_0.subtask_2, construct the one-loop radiative correction expressions "
        "for the pseudo-Goldstone boson mass squared M_h2^2. For each particle with nonzero coupling, determine the sign, coefficient alpha_i (including multiplicity and coupling strength), "
        "and mass dependence. Ensure dimensional consistency and physical correctness of each term. Avoid including particles without verified couplings."
    )
    cot_sc_desc_1_1 = {
        'instruction': cot_sc_instruction_1_1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent one-loop radiative correction expression.",
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Integrate and refine the candidate mass formulae by critically evaluating the completeness of bosonic and fermionic contributions, "
        "the role of vacuum expectation values (x, v), and the placement of the symmetry breaking scale factor (numerator vs denominator). Explicitly analyze the physical origin and necessity of each alpha_i coefficient, "
        "including differences in singlet fermion contributions. This subtask must embed a structured constrained chain-of-thought to prevent premature assumptions and ensure all terms are justified."
    )
    critic_instruction_1_2 = (
        "Please review and provide the limitations of provided solutions of the candidate mass formulae, focusing on physical consistency, dimensional correctness, and completeness of contributions."
    )
    cot_reflect_desc_1_2 = {
        'instruction': cot_reflect_instruction_1_2,
        'critic_instruction': critic_instruction_1_2,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Evaluate the refined candidate mass formulae against theoretical consistency, dimensional analysis, and physical completeness criteria. "
        "Conduct a rigorous debate among agents to challenge assumptions such as omission of the pseudoscalar A0 term or incorrect sign assignments. "
        "Explicitly justify inclusion/exclusion of each term and confirm the final formula aligns with the model-specific couplings and known radiative correction structures."
    )
    final_decision_instruction_2_1 = "Sub-task 1: Finalize and confirm the most theoretically consistent and physically complete mass formula for the pseudo-Goldstone boson."
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])

    return final_answer, logs
