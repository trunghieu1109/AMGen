async def forward_150(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze and classify the given system state vector and observable operator matrix, "
        "including verifying normalization of the state vector and Hermiticity of the operator matrix, "
        "with context from the user query."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing system state and operator, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_1_1 = (
        "Sub-task 1: Determine the eigenvalues and eigenvectors of the observable operator P, "
        "identifying the eigenspace corresponding to eigenvalue 0, with context from previous analysis. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_1_1,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_1_1, answer_1_1 = await agent([taskInfo, thinking_0_1, answer_0_1], debate_instr_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking_0_1, answer_0_1] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking_1_1, answer_1_1 = await agent(input_infos_1_1, debate_instr_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_decision_instr_1_1 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_1, answer_0_1] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1: Determine eigenvalues and eigenvectors of P. " + final_decision_instr_1_1,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Normalize the given system state vector to ensure it is a valid quantum state, "
        "based on the analysis from stage_0.subtask_1."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_1_2)
    ]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, normalizing state vector, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_1_2 + possible_answers_1_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct normalization of the state vector.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    agents.append(f"Final Decision agent, normalizing state vector, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Project the normalized state vector onto the eigenspace of eigenvalue 0 and compute the squared magnitude "
        "of this projection to find the measurement probability, using outputs from stage_1.subtask_1 and stage_1.subtask_2."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_2_1)
    ]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i](
            [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
            cot_sc_instruction_2_1, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, projecting and computing probability, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + possible_thinkings_2_1 + possible_answers_2_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct measurement probability.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    agents.append(f"Final Decision agent, computing measurement probability, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
