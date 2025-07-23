async def forward_183(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze the target molecule's substitution pattern and determine the required functional groups and their relative positions on the benzene ring, including the directing effects of each substituent, with context from the query."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent analysis of the target molecule's substitution pattern."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ['user query']
    }
    results1, log1 = await self.sc_cot(
        subtask_id='subtask_1',
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, map each reagent and condition in the given options to their corresponding chemical transformations and understand their typical regioselectivity and compatibility in aromatic substitution sequences."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent mapping of reagents to transformations and regioselectivity."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate each provided reaction sequence option step-by-step to predict the intermediate products, considering directing effects, possible side reactions, and the feasibility of each transformation in the given order, using the analyses from Subtasks 1 and 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Debate and decide the most feasible and chemically sound evaluation of the reaction sequences."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2'],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Refine the analysis by comparing the predicted outcomes of each sequence, focusing on yield, regioselectivity, and practicality to identify the most plausible high-yield synthetic route to the target compound."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of the provided solutions of the reaction sequence evaluations, focusing on yield, regioselectivity, and practicality."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results4, log4 = await self.reflexion(
        subtask_id='subtask_4',
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_agent_instruction5 = (
        "Sub-task 5: Generate a final recommendation of the optimal reaction sequence from the given options, supported by mechanistic reasoning and synthesis strategy considerations, based on the refined analysis from Subtask 4."
    )
    cot_agent_desc5 = {
        'instruction': cot_agent_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4']
    }
    results5, log5 = await self.cot(
        subtask_id='subtask_5',
        cot_agent_desc=cot_agent_desc5
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
