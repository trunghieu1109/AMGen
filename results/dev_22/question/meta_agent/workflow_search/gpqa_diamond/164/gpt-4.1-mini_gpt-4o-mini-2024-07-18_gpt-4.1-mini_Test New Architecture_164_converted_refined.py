async def forward_164(self, taskInfo):
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Critically evaluate the claim that dual homogeneous catalyst systems for ethylene polymerization producing regularly branched polyethylene "
        "are implemented on an industrial scale in the US. Perform targeted literature and patent search to verify industrial practice, identify counterexamples or limitations, "
        "and avoid overgeneralization by cross-validating with industrial data and expert reviews, based on the provided query."
    )
    debate_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_1, log_1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_1)

    cot_instruction_2 = (
        "Sub-task 2: Assess the chemical compatibility and functionality of aluminum-based activators in the essential additional reaction step for introducing regular branches in polyethylene from ethylene alone. "
        "Perform mechanistic analysis and literature validation to confirm or refute the claim that aluminum-based activators do not work, considering known catalyst activation systems such as MAO and their role in chain-walking or branching mechanisms, based on the provided query."
    )
    debate_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_2, log_2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc_2,
        n_repeat=self.max_round
    )
    logs.append(log_2)

    cot_instruction_3 = (
        "Sub-task 3: Evaluate the feasibility of using group VIa transition metal catalysts (e.g., Cr, Mo, W) with specific activators to produce regularly branched polyethylene from ethylene alone. "
        "Include mechanistic scrutiny of polymerization pathways, branching control, and literature/patent evidence on catalyst performance and limitations. Identify any mechanistic or industrial counterexamples that challenge the statement's validity, based on the provided query."
    )
    debate_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_3, log_3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc_3,
        n_repeat=self.max_round
    )
    logs.append(log_3)

    cot_instruction_4 = (
        "Sub-task 4: Analyze the use and economic considerations of noble metal catalysts for the branching reaction in ethylene polymerization. "
        "Review catalyst cost, performance, and industrial applicability, focusing on whether noble metal catalysts can achieve the desired branching but are prohibitively expensive. "
        "Validate with economic data and industrial reports, and identify if this is the uniquely correct statement among the four, based on the provided query."
    )
    debate_desc_4 = {
        'instruction': cot_instruction_4,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_4, log_4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        n_repeat=self.max_round
    )
    logs.append(log_4)

    cot_sc_instruction_5 = (
        "Sub-task 5: Perform a comprehensive literature and patent search focused on dual homogeneous catalyst systems for ethylene polymerization with branching, activator types, and catalyst classes mentioned in the statements. "
        "Provide an evidence-based foundation to support or refute each statement, preventing blind acceptance and ensuring mechanistic and industrial reality are considered, based on the provided query."
    )
    results_5, log_5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc={
            'instruction': cot_sc_instruction_5,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log_5)

    cot_sc_instruction_6 = (
        "Sub-task 6: Integrate findings from subtasks 1 through 5 to critically evaluate the exclusivity, chemical feasibility, industrial implementation, and economic factors of each statement. "
        "Explicitly identify contradictions, overlaps, and unique correctness. Select the single most correct statement regarding the formation of regularly branched polyethylene using only ethylene and a dual catalyst system, avoiding overgeneralization and ensuring coherence with expert feedback, based on the provided query and previous subtasks' outputs."
    )
    results_6, log_6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_agent_desc={
            'instruction': cot_sc_instruction_6,
            'input': [taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer'], results_3['thinking'], results_3['answer'], results_4['thinking'], results_4['answer'], results_5['thinking'], results_5['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"]
        }
    )
    logs.append(log_6)

    cot_reflect_instruction_7 = (
        "Sub-task 7: Conduct a final reflexive review of the integrated conclusion from subtask 6 to ensure the reasoning is robust, the selected statement is uniquely justified, "
        "and no prior errors (such as accepting all statements or ignoring mechanistic evidence) are repeated. Explicitly check for consistency with industrial practice, chemical mechanisms, and economic realities, "
        "and confirm the final answer aligns with expert feedback, based on the provided query and previous subtasks' outputs."
    )
    results_7, log_7 = await self.reflexion(
        subtask_id="subtask_7",
        reflect_desc={
            'instruction': cot_reflect_instruction_7,
            'input': [taskInfo, results_6['thinking'], results_6['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        },
        n_repeat=self.max_round
    )
    logs.append(log_7)

    final_answer = await self.make_final_answer(results_7['thinking'], results_7['answer'])
    return final_answer, logs
