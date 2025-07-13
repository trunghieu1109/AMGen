async def forward_184(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = "Sub-task 1: Extract and summarize the defining features of the Hamiltonian operator H = epsilon sigma · n, including the nature of sigma (Pauli matrices), the unit vector n, and the constant epsilon, with context from the user query."
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting and summarizing Hamiltonian features, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_1_1 = "Sub-task 1: Analyze and classify the spectral properties of the operator sigma · n, focusing on its eigenvalues and their physical meaning, given the summary from stage_0.subtask_1. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_1 = []
    all_answer_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1_1):
        thinking_i, answer_i = await agent([taskInfo, thinking_0, answer_0], debate_instruction_1_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, analyzing spectral properties, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_1_1.append(thinking_i)
        all_answer_1_1.append(answer_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0, answer_0] + all_thinking_1_1 + all_answer_1_1, "Sub-task 1: Analyze spectral properties of sigma · n. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, analyzing spectral properties, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = "Sub-task 2: Clarify the role of physical constants such as hbar and their presence or absence in the Hamiltonian and eigenvalues, resolving ambiguities in the problem statement, based on the output from stage_0.subtask_1."
    N_sc = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, clarifying role of hbar, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0, answer_0] + possible_thinkings_1_2 + possible_answers_1_2, "Sub-task 2: Clarify role of physical constants. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, clarifying hbar role, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 1: Transform the spectral properties of sigma · n into explicit eigenvalue expressions for the Hamiltonian H by scaling with epsilon and considering the role of hbar if relevant, based on outputs from stage_1.subtask_1 and stage_1.subtask_2. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_2_1 = []
    all_answer_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_2_1):
        thinking_i, answer_i = await agent([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], debate_instruction_2_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, transforming spectral properties to eigenvalues, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_2_1.append(thinking_i)
        all_answer_2_1.append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_2_1 + all_answer_2_1, "Sub-task 1: Transform spectral properties into explicit eigenvalues. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, transforming spectral properties, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    debate_instruction_3_1 = "Sub-task 1: Evaluate the given multiple-choice eigenvalue options against the derived eigenvalues from stage_2.subtask_1 to identify the correct answer. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3_1 = []
    all_answer_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3_1):
        thinking_i, answer_i = await agent([taskInfo, thinking_2_1, answer_2_1], debate_instruction_3_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, evaluating multiple-choice options, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_3_1.append(thinking_i)
        all_answer_3_1.append(answer_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo, thinking_2_1, answer_2_1] + all_thinking_3_1 + all_answer_3_1, "Sub-task 1: Evaluate multiple-choice eigenvalue options. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, evaluating multiple-choice options, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
