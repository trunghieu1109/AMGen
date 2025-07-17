async def forward_164(self, taskInfo):
    logs = []

    cot_instruction_1 = "Sub-task 1: Evaluate the industrial implementation claim: verify if dual catalyst systems for ethylene polymerization with branching are indeed implemented on an industrial scale in the US, based on the provided query and chemical industry knowledge."
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "industrial polymerization", "dual catalyst systems"]
    }
    results_1, log_1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_1)

    cot_instruction_2 = "Sub-task 2: Assess the chemical compatibility and functionality of aluminum-based activators in the essential additional reaction step for branching in ethylene polymerization, based on the provided query and catalysis knowledge."
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "organometallic catalysis", "activators"]
    }
    results_2, log_2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc_2,
        n_repeat=self.max_round
    )
    logs.append(log_2)

    cot_instruction_3 = "Sub-task 3: Evaluate the feasibility of using group VIa transition metal catalysts with specific activators for producing branched polyethylene from ethylene alone, based on the provided query and catalysis principles."
    cot_agent_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "group VIa catalysts", "polymer branching"]
    }
    results_3, log_3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_agent_desc_3,
        n_repeat=self.max_round
    )
    logs.append(log_3)

    cot_instruction_4 = "Sub-task 4: Analyze the use and economic considerations of noble metal catalysts for the branching reaction in ethylene polymerization, based on the provided query and industrial cost factors."
    cot_agent_desc_4 = {
        'instruction': cot_instruction_4,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "noble metal catalysts", "economic analysis"]
    }
    results_4, log_4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=cot_agent_desc_4,
        n_repeat=self.max_round
    )
    logs.append(log_4)

    cot_sc_instruction_5 = "Sub-task 5: Integrate chemical, industrial, and economic analyses from Stage 0 subtasks to compute the conditional validity of each statement regarding catalyst systems for branched polyethylene production."
    cot_sc_desc_5 = {
        'instruction': cot_sc_instruction_5,
        'input': [taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer'], results_3['thinking'], results_3['answer'], results_4['thinking'], results_4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results_5, log_5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc_5,
        n_repeat=self.max_sc
    )
    logs.append(log_5)

    debate_instruction_6 = "Sub-task 6: Evaluate and determine which of the four statements is correct based on the integrated analysis, ensuring coherence with industrial practice, chemical feasibility, and economic factors."
    debate_desc_6 = {
        'instruction': debate_instruction_6,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'input': [taskInfo, results_5['thinking'], results_5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_6, log_6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc_6,
        n_repeat=self.max_round
    )
    logs.append(log_6)

    final_answer = await self.make_final_answer(results_6['thinking'], results_6['answer'])
    return final_answer, logs
