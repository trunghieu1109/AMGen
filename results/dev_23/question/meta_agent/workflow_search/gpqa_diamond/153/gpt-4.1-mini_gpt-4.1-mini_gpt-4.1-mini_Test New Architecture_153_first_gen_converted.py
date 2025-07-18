async def forward_153(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize quantitative spectral data from mass spectrometry, IR, and 1H NMR to identify molecular weight, isotopic pattern, functional groups, and proton environments, "
        "with context from the provided query about an unidentified drug spectral data."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="subtask_1_stage_0",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    debate_instruction1 = (
        "Sub-task 1: Integrate and interpret the summarized spectral data from Sub-task 1 of Stage 0 to deduce key structural features such as presence of chlorine, carboxylic acid group, and aromatic substitution pattern, "
        "debating the evidence and reasoning with context from the query and previous outputs."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'context': ["user query", results0_1['thinking'], results0_1['answer']],
        'input': [taskInfo, results0_1['thinking'], results0_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="subtask_1_stage_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    reflexion_instruction2 = (
        "Sub-task 2: Compare the deduced structural features from Sub-task 1 of Stage 1 with the given candidate compounds to evaluate which structure best fits the spectral evidence, "
        "reflecting on the validity and limitations of each candidate with context from the query and previous outputs."
    )
    critic_instruction2 = (
        "Please review the candidate evaluation and provide its limitations and confidence level."
    )
    reflexion_desc2 = {
        'instruction': reflexion_instruction2,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']]
    }
    results1_2, log1_2 = await self.reflexion(
        subtask_id="subtask_2_stage_1",
        reflect_desc=reflexion_desc2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction3 = (
        "Sub-task 3: Make a reasoned structural suggestion for the unidentified compound based on the integrated spectral analysis and candidate evaluation from previous subtasks, "
        "debating the final choice with context from the query and all previous outputs."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'input': [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results1_3, log1_3 = await self.debate(
        subtask_id="subtask_3_stage_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log1_3)

    final_answer = await self.make_final_answer(results1_3['thinking'], results1_3['answer'])
    return final_answer, logs
