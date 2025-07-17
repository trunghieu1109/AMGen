async def forward_170(self, taskInfo):
    logs = []

    cot_instruction0 = (
        "Sub-task 0: Extract and summarize the essential chemical information about the six substances, "
        "their substituents, and the reaction conditions to define the problem context clearly. "
        "Focus on the given benzene derivatives and the electrophilic substitution with bromine."
    )
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0, log0 = await self.cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_agent_desc0
    )
    logs.append(log0)

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze and classify the substituents on the benzene ring according to their electronic effects "
        "(electron-donating or withdrawing) and directing influence (ortho/para or meta) on electrophilic substitution, "
        "based on the summary from Sub-task 0."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 0", "answer of subtask 0"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Transform the classification of substituents from Sub-task 1 into predicted regioselectivity outcomes, "
        "estimating the relative para-isomer yields based on directing effects and steric considerations. "
        "Engage in a debate to weigh different chemical reasoning perspectives."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"],
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

    debate_instruction3 = (
        "Sub-task 3: Evaluate and prioritize the six substances by arranging them in order of increasing weight fraction of the para-isomer yield, "
        "using the predicted regioselectivity and chemical reasoning from Sub-task 2. "
        "Conduct a debate to finalize the ranking."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
