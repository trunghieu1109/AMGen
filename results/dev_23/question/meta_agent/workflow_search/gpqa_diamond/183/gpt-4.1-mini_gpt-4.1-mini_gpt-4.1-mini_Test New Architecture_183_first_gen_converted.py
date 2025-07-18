async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the target molecule's substitution pattern and identify the required functional groups and their positions on the benzene ring, with context from the query about synthesizing 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene."
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

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, map each reagent and reaction step in the given options to their chemical transformations and expected regioselective effects on benzene, considering the target molecule's substitution pattern."
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Evaluate the compatibility and order of reaction steps in each sequence to ensure correct regioselectivity and functional group transformations leading to the target molecule, debating the merits and drawbacks of each sequence."
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc={
            'instruction': debate_instruction3,
            'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = "Sub-task 4: Based on the outputs from Sub-task 3, assess the potential yield and practicality of each reaction sequence considering known chemical principles, side reactions, and isomer formation."
    critic_instruction4 = "Please review the assessment of potential yield and practicality and provide its limitations."
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc={
            'instruction': cot_reflect_instruction4,
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']]
        },
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Based on the output of Sub-task 4, select the optimal sequence of reactions from the given options that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene."
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc={
            'instruction': debate_instruction5,
            'context': ["user query", results4['thinking'], results4['answer']],
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
