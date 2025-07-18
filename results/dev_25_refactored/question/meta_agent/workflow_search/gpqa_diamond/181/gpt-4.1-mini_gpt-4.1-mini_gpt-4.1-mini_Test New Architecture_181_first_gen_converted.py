async def forward_181(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and summarize the physical assumptions and conditions underlying the Mott-Gurney equation, "
        "including the role of trap states, carrier type, contact nature, and current components (drift vs diffusion), "
        "with context from the user query."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Integrate the summarized physical assumptions from Sub-task 1 with the characteristics described in each choice, "
        "to map which conditions align with the Mott-Gurney equation's validity."
    )
    critic_instruction2 = (
        "Please review and provide the limitations of the integration of physical assumptions with the choices, "
        "highlighting any inconsistencies or clarifications needed."
    )
    cot_reflect_desc2 = {
        "instruction": cot_reflect_instruction2,
        "critic_instruction": critic_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate each choice against the integrated criteria from Sub-task 2 to identify which statement correctly describes the validity conditions of the Mott-Gurney equation."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the final answer identifying the correct statement about the validity of the Mott-Gurney equation."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
