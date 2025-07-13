async def forward_177(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Identify and confirm the standard mass dimensions of the fields psi, F^{mu nu}, "
        "and the operator sigma_{mu nu} in 4D spacetime, given the interaction Lagrangian and context."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, identifying mass dimensions, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 1: Based on the identified mass dimensions of psi, F^{mu nu}, and sigma_{mu nu}, "
        "compute the mass dimension of the coupling constant kappa by ensuring the interaction Lagrangian term has total mass dimension 4."
    )
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, computing mass dimension of kappa, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent and correct mass dimension of kappa."
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo, thinking_0, answer_0] + possible_thinkings_1 + possible_answers_1,
        "Sub-task stage_1.subtask_1: Synthesize and choose the most consistent mass dimension of kappa." + final_instr_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Determine the renormalizability of the theory based on the mass dimension of kappa "
        "and standard renormalization criteria, using the mass dimension result from previous subtask."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, determining renormalizability, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_1 = "Given all the above thinking and answers, find the most consistent and correct renormalizability conclusion."
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking_1, answer_1] + possible_thinkings_2_1 + possible_answers_2_1,
        "Sub-task stage_2.subtask_1: Synthesize and choose the most consistent renormalizability conclusion." + final_instr_2_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Combine the results of the mass dimension and renormalizability analysis to select the correct answer choice "
        "from the provided options, using the outputs from previous subtasks."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_2, answer_2_2 = await cot_agents_2_2[i](
            [taskInfo, thinking_1, answer_1, thinking_2_1, answer_2_1],
            cot_sc_instruction_2_2,
            is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, selecting final answer choice, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_2 = "Given all the above thinking and answers, find the most consistent and correct final answer choice."
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2(
        [taskInfo, thinking_1, answer_1, thinking_2_1, answer_2_1] + possible_thinkings_2_2 + possible_answers_2_2,
        "Sub-task stage_2.subtask_2: Synthesize and choose the most consistent final answer choice." + final_instr_2_2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
