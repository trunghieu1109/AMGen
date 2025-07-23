async def forward_188(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and categorize relevant information about each particle (Magnon, Skyrmion, Pion, Phonon) "
        "and the concept of spontaneous symmetry breaking from the query and background knowledge."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Classify each particle (Magnon, Skyrmion, Pion, Phonon) as either associated or not associated "
        "with spontaneously-broken symmetry based on their physical origin and theoretical interpretation, "
        "using the extracted information from Sub-task 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent classification for each particle regarding their association "
        "with spontaneously-broken symmetry."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Filter and select the particle(s) that are not associated with spontaneously-broken symmetry "
        "from the classified list obtained in Sub-task 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the final answer identifying the particle(s) not associated with spontaneously-broken symmetry."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
