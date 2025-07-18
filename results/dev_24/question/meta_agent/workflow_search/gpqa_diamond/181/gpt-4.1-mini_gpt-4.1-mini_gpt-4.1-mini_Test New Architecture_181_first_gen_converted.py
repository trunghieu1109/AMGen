async def forward_181(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the fundamental physical assumptions and device characteristics underlying the Mott-Gurney equation, "
        "including device type (single- or two-carrier), trap presence, contact type (Ohmic or Schottky), carrier injection barriers, and current components (drift vs diffusion). "
        "Use the provided query and choices as context."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Evaluate and prioritize the given statements by comparing each against the classified assumptions and conditions "
        "identified in Sub-task 1 to determine their consistency with the theoretical validity of the Mott-Gurney equation. "
        "Engage in a debate to weigh pros and cons of each choice."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct statement about the validity of the Mott-Gurney equation, "
        "based on the debate and analysis from Sub-task 1."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the final determination of which statement is true about the validity of the Mott-Gurney equation by synthesizing the analysis and evaluation results "
        "from Sub-tasks 1 and 2, and provide a clear justification."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the final answer selecting the true statement about the validity of the Mott-Gurney equation, "
        "with clear justification based on previous subtasks."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
