async def forward_150(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze and classify the given system state vector and the observable matrix operator P, "
        "including checking normalization of the state vector and confirming properties of P (e.g., Hermiticity). "
        "Given the user query: " + taskInfo['question']
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing state vector and operator P, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_1_1 = (
        "Sub-task 1: Compute the eigenvalues and eigenvectors of the observable matrix operator P, "
        "and identify the eigenspace corresponding to the eigenvalue 0. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. "
        "User query: " + taskInfo['question']
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
            agents.append(f"Debate agent {agent.id}, round {r}, computing eigenvalues/vectors, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_decision_instr_1_1 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_1, answer_0_1] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task stage_1.subtask_1: Compute eigenvalues and eigenvectors and identify eigenspace for eigenvalue 0. " + final_decision_instr_1_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Normalize the system state vector if not already normalized, preparing it for projection calculations. "
        "User query: " + taskInfo['question'] + ". Use output from stage_0.subtask_1."
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
        "Sub-task stage_1.subtask_2: Normalize the state vector. Choose the most consistent and correct solution.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Project the normalized state vector onto the eigenspace corresponding to eigenvalue 0 and compute the probability of measuring 0 by calculating the squared magnitude of this projection. "
        "Use outputs from stage_1.subtask_1 and stage_1.subtask_2. "
        "User query: " + taskInfo['question']
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
        "Sub-task stage_2.subtask_1: Compute final probability of measuring eigenvalue 0. Choose the most consistent and correct solution.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
