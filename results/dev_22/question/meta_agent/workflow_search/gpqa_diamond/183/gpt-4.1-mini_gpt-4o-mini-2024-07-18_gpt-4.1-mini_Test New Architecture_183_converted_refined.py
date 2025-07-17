async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each reaction step in the given sequences by identifying their chemical role, "
        "reaction type, and expected effect on the benzene ring. Include detailed annotations of reagents and transformations "
        "to support regioselectivity analysis, based on the query: 'Which sequence of reactions from the following options would lead to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, beginning with benzene?'."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Perform explicit structural mapping of substituents on the benzene ring after each reaction step in every sequence. "
        "Assign ring positions (1 to 6) and annotate substituents at each stage. Evaluate directing effects and regioselectivity, "
        "verifying alignment with the target compound 2-(tert-butyl)-1-ethoxy-3-nitrobenzene. Identify regioselectivity conflicts or invalid steps, "
        "using outputs from Sub-task 1."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    reflexion_instruction3 = (
        "Sub-task 3: Assess the compatibility and order of reaction steps in each sequence based on the detailed substitution mapping and directing effects from Sub-task 2. "
        "Verify functional group transformations proceed without interference and the sequence logically leads to the desired substitution pattern. "
        "Re-examine assumptions about reaction feasibility, steric effects, and sulfonation/desulfonation timing."
    )
    critic_instruction3 = (
        "Please review the validity and limitations of the compatibility assessment for each sequence."
    )
    cot_reflect_desc3 = {
        'instruction': reflexion_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    reflexion_instruction4 = (
        "Sub-task 4: Compare the four given sequences using verified substitution maps and compatibility assessments from Sub-task 3. "
        "Identify which sequence most likely yields 2-(tert-butyl)-1-ethoxy-3-nitrobenzene in high yield by evaluating regioselectivity, directing effects, reaction order, and functional group transformations. "
        "Incorporate detailed structural bookkeeping and mechanistic insights to avoid past errors."
    )
    critic_instruction4 = (
        "Please review the comparison and selection of the optimal sequence, noting any limitations or uncertainties."
    )
    cot_reflect_desc4 = {
        'instruction': reflexion_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_instruction5 = (
        "Sub-task 5: Summarize the reasoning process and select the optimal reaction sequence. Provide detailed justification based on explicit mapping of substituent positions, directing effects, reaction order, and functional group transformations. "
        "Highlight how the chosen sequence overcomes pitfalls and ensures correct substitution pattern for high-yield synthesis of the target compound."
    )
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
