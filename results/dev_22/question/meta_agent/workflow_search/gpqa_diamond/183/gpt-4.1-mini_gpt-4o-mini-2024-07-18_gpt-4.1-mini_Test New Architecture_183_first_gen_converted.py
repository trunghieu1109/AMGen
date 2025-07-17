async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each reaction step in the given sequences by identifying their chemical role, "
        "reaction type, and expected effect on the benzene ring, with context from the provided query."
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
        "Sub-task 2: Evaluate the directing effects and regioselectivity of substituents introduced at each step "
        "to predict substitution patterns and positions on the benzene ring, based on outputs from Sub-task 1."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Assess the compatibility and order of reaction steps in each sequence to ensure functional group "
        "transformations proceed without interference and lead to the desired substitution pattern, using outputs from Sub-tasks 1 and 2."
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

    debate_instruction4 = (
        "Sub-task 4: Compare the four given sequences based on the predicted regioselectivity, directing effects, "
        "and reaction compatibility to identify which sequence most likely yields 2-(tert-butyl)-1-ethoxy-3-nitrobenzene in high yield, using outputs from Sub-task 3."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_instruction5 = (
        "Sub-task 5: Summarize the reasoning and select the optimal reaction sequence, providing justification based on "
        "the analysis of directing effects, reaction order, and functional group transformations, using outputs from Sub-task 4."
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
