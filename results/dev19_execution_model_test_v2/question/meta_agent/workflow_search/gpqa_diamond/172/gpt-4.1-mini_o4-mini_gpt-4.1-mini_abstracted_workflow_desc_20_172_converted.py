async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive the formula and theoretical framework to estimate the minimum uncertainty in energy ΔE "
        "from the given uncertainty in position Δx and electron velocity v, using the Heisenberg uncertainty principle and kinetic energy relations."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving theoretical framework, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Calculate the minimum uncertainty in momentum Δp using the Heisenberg uncertainty principle Δx · Δp ≥ ħ/2 with the given Δx."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, calculating Δp, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_1 = "Given all the above thinking and answers, find the most consistent and correct estimate for minimum uncertainty in momentum Δp."
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0, answer_0] + possible_thinkings_1_1 + possible_answers_1_1,
        "Stage 1 Sub-task 1: Calculate minimum uncertainty in momentum Δp." + final_instr_1_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Estimate the minimum uncertainty in energy ΔE from the calculated Δp, considering the kinetic energy relation appropriate for the electron's speed (evaluate if relativistic corrections are needed)."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0, answer_0, thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_2[i](
            [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1],
            cot_sc_instruction_1_2, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, estimating ΔE from Δp, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_2 = "Given all the above thinking and answers, find the most consistent and correct estimate for minimum uncertainty in energy ΔE."
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1] + possible_thinkings_1_2 + possible_answers_1_2,
        "Stage 1 Sub-task 2: Estimate minimum uncertainty in energy ΔE." + final_instr_1_2,
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    reflect_inst_2_1 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Assess the impact of relativistic effects on the uncertainty in energy ΔE and verify if the non-relativistic approximation is valid or if relativistic formulas must be applied. "
        + reflect_inst_2_1
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, assessing relativistic effects, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_reflect):
        critic_inst_2_1 = (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback_2_1, correct_2_1 = await critic_agent_2_1(
            [taskInfo, thinking_2_1, answer_2_1],
            "Stage 2 Sub-task 1: Criticize relativistic assessment." + critic_inst_2_1,
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback on relativistic assessment, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining relativistic assessment, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Integrate all findings to finalize the estimate of the minimum uncertainty in energy ΔE and select the closest matching choice from the given options."
    )
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, thinking_2_1, answer_2_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_2_2[i](
            [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, thinking_2_1, answer_2_1],
            cot_sc_instruction_2_2, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, finalizing ΔE estimate and choice selection, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)

    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_2 = "Given all the above thinking and answers, find the most consistent and correct final estimate of ΔE and select the closest choice."
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2(
        [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, thinking_2_1, answer_2_1] + possible_thinkings_2_2 + possible_answers_2_2,
        "Stage 2 Sub-task 2: Finalize ΔE estimate and choice selection." + final_instr_2_2,
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
