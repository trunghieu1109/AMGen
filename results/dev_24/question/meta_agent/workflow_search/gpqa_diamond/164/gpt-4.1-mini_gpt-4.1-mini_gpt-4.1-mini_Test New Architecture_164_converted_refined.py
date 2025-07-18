async def forward_164(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and classify key elements from the query, including catalyst types (group VIa transition metals, noble metals), "
        "activators (aluminum-based and others), polymerization goals (high-density polymer, branched polymer), and industrial context (US implementation). "
        "Avoid assumptions by strictly summarizing only given information without inferring industrial status or chemical compatibility."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the chemical and economic relationships and constraints between catalysts, activators, and polymer branching, "
        "focusing on catalyst-activator compatibility, polymer microstructure control, and cost considerations. "
        "Explicitly separate chemical feasibility from economic feasibility to avoid conflation."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent analysis for chemical and economic relationships given the output from Sub-task 1. "
        "Find the most consistent and correct conclusions for catalyst and activator compatibility and cost considerations."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
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
        "Sub-task 3: Critically verify the factual claims in each of the four statements with evidence or authoritative sources, "
        "especially regarding industrial implementation status and activator effectiveness. "
        "Explicitly challenge broad claims and chemical compatibility assertions. Avoid accepting statements based on assumptions or consensus without evidence."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a reasoned judgment on the factual correctness of each statement independently, "
        "based on verified evidence and chemical-economic analysis."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate each statement independently based on the verified evidence and chemical-economic analysis from Sub-task 3. "
        "Identify which statement(s) are factually and chemically correct, ensuring no conflation of conditional or partial truths. "
        "Document reasoning for acceptance or rejection of each statement."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide a detailed evaluation and final verdict on each statement's correctness."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Re-examine and challenge prior conclusions, especially focusing on the industrial implementation claim and activator compatibility. "
        "Synthesize the final single correct statement by integrating chemical facts, industrial realities, and economic considerations, "
        "explicitly overturning any previously uncritically accepted claims. Ensure alignment with expert feedback and avoid repeating past errors."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions for this problem, focusing on potential errors or overlooked aspects."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
