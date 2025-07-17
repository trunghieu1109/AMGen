async def forward_182(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze and classify the structural elements and unsaturation of the starting compound 2-formyl-5-vinylcyclohex-3-enecarboxylic acid, including ring, double bonds, and functional groups relevant to IHD calculation, with context from taskInfo"
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

    debate_instruction2 = "Sub-task 2: Determine the chemical transformations and structural changes induced by red phosphorus and excess HI on each functional group (formyl, vinyl, carboxylic acid) and the ring system to predict the product structure, based on output from Sub-task 1"
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", results1['thinking'], results1['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = "Sub-task 3: Calculate the index of hydrogen deficiency (IHD) of the predicted product structure based on the transformations identified in Sub-task 2 and the initial classification from Sub-task 1"
    results1_thinking = results1.get('thinking', '')
    results1_answer = results1.get('answer', '')
    results2_thinking = results2.get('thinking', '')
    results2_answer = results2.get('answer', '')

    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1_thinking, results1_answer, results2_thinking, results2_answer],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Select the correct IHD value of the product from the given choices (0, 1, 3, 5) based on the calculated IHD from Sub-task 3"
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
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
