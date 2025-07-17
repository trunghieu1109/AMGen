async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each reagent and reaction step in the given options according to their chemical function (e.g., Friedel-Crafts alkylation, nitration, reduction, diazotization, sulfonation, hydrolysis, nucleophilic substitution) and their typical regioselectivity and directing effects on benzene. "
        "Explicitly identify the directing nature (ortho/para vs. meta) of substituents introduced and potential reaction compatibility issues, embedding the failure reason that previous reasoning oversimplified directing effects and ignored reaction incompatibilities."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Perform a detailed mechanistic feasibility analysis focusing on: (1) the directing strengths and positional outcomes of tert-butyl vs. nitro substituents, "
        "(2) the stability and compatibility of tert-butyl groups under nitration conditions, (3) the necessity and role of sulfonation/desulfonation as blocking/deblocking steps to control regioselectivity, "
        "and (4) the correct order and mechanism of phenol-to-ether conversion via reduction, diazotization, hydrolysis, and Williamson ether synthesis. "
        "This subtask explicitly addresses the failure reasons that previous workflows assumed direct nitration after alkylation without blocking and misunderstood the ether formation mechanism."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate each provided reaction sequence option against the target molecule’s substitution pattern and yield considerations, "
        "integrating mechanistic insights from subtasks 1 and 2. Identify which sequence logically and efficiently leads to 2-(tert-butyl)-1-ethoxy-3-nitrobenzene with high yield, "
        "explicitly avoiding the flawed assumptions of direct nitration after alkylation and ignoring blocking strategies."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="stage_2.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Identify and explain the critical role of sulfonation/desulfonation steps and the order of nitration and alkylation in controlling regioselectivity and preventing side reactions in the synthesis sequence. "
        "This subtask must explicitly incorporate the feedback that blocking strategies are essential and that ignoring them leads to poor regioselectivity and yield."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_2.subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Select the optimal reaction sequence from the given options based on the comprehensive analysis and evaluation from previous subtasks. "
        "Provide a clear, mechanistically justified rationale for why this sequence leads to the high-yield synthesis of the target compound starting from benzene, "
        "explicitly referencing the importance of directing effects, blocking strategies, and correct order of functional group transformations to avoid the errors of prior reasoning."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3", "thinking of stage_2.subtask_4", "answer of stage_2.subtask_4"]
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
