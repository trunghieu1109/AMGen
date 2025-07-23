async def forward_185(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and categorize all relevant chemical information from the query, "
        "including the structure and stereochemistry of the starting material, the nature of the Cope rearrangement, "
        "and the characteristics of the product choices."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent extraction and categorization of chemical information from the query."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Analyze the mechanistic relationships and stereochemical constraints of the Cope rearrangement "
        "on the given azabicyclic system to predict possible rearrangement pathways and product frameworks."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Select the most plausible mechanistic pathway and stereochemical outcome consistent with the starting material and reaction type."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Derive the possible product structures from the Cope rearrangement mechanism "
        "and validate these structures against the given product names and their hydrogenation patterns."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent product structures matching the mechanism and product names."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate all candidate products based on mechanistic feasibility, stereochemical consistency, "
        "and structural correctness to select the best matching product for the given reaction."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best matching product from the candidates based on all chemical and mechanistic evidence."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
