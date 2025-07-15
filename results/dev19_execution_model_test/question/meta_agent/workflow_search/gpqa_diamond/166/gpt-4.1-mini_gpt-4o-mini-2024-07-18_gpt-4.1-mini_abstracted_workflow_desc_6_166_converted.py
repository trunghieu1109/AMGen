async def forward_166(self, taskInfo):
    from collections import Counter
    import numpy as np
    import scipy.linalg
    import math
    sub_tasks = []
    agents = []
    logs = []

    phi = -np.pi / 4
    alpha = 0.5

    # Stage 1: Construct states
    cot_sc_instruction_1 = (
        "Sub-task 1: Construct the normalized Schrödinger cat state |psi> for phi = -pi/4 and alpha = 0.5. "
        "Calculate normalization constant N, write explicit state vector in truncated Fock basis (dimension=10), and construct density matrix rho. "
        "Use coherent states expansion in Fock basis and ensure normalization and phase conventions are clear."
    )
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, constructing Schrödinger cat state, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent construction of Schrödinger cat state." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the Schrödinger cat state density matrix rho from Sub-task 1, "
        "construct the reference Gaussian state tau that matches the first and second moments of rho. "
        "Compute covariance matrix explicitly and represent tau as a density matrix in the same truncated basis. "
        "Provide explicit formulas and numerical matrices."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, constructing reference Gaussian state, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent construction of reference Gaussian state." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Eigen-decomposition and trace computations
    debate_instruction_3a = (
        "Sub-task 3a: Perform eigenvalue decomposition of the density matrix rho from Sub-task 1. "
        "Verify positive semidefinite and normalization. Provide eigenvalues and eigenvectors with numerical precision."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking_3a = [[] for _ in range(N_max_3a)]
    all_answer_3a = [[] for _ in range(N_max_3a)]
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instruction_3a,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking3a, answer3a = await agent([taskInfo, thinking1, answer1], debate_instruction_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking1, answer1] + all_thinking_3a[r-1] + all_answer_3a[r-1]
                thinking3a, answer3a = await agent(input_infos_3a, debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, eigen-decomposition of rho, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking_3a[r].append(thinking3a)
            all_answer_3a[r].append(answer3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo, thinking1, answer1] + all_thinking_3a[-1] + all_answer_3a[-1], "Sub-task 3a: Final decision on eigen-decomposition of rho." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])

    debate_instruction_3b = (
        "Sub-task 3b: Compute trace Tr(rho ln rho) using eigenvalues from Sub-task 3a. "
        "Handle zero eigenvalues carefully and provide intermediate numerical results. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking_3b = [[] for _ in range(N_max_3b)]
    all_answer_3b = [[] for _ in range(N_max_3b)]
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": debate_instruction_3b,
        "context": ["user query", thinking3a, answer3a],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], debate_instruction_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking_3b[r-1] + all_answer_3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, debate_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compute Tr(rho ln rho), thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking_3b[r].append(thinking3b)
            all_answer_3b[r].append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo, thinking3a, answer3a] + all_thinking_3b[-1] + all_answer_3b[-1], "Sub-task 3b: Final decision on Tr(rho ln rho)." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])

    debate_instruction_3c = (
        "Sub-task 3c: Perform eigenvalue decomposition of the reference Gaussian state density matrix tau from Sub-task 2. "
        "Verify normalization and positivity. Provide eigenvalues and eigenvectors numerically. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3c = self.max_round
    all_thinking_3c = [[] for _ in range(N_max_3c)]
    all_answer_3c = [[] for _ in range(N_max_3c)]
    subtask_desc_3c = {
        "subtask_id": "subtask_3c",
        "instruction": debate_instruction_3c,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3c):
        for i, agent in enumerate(debate_agents_3c):
            if r == 0:
                thinking3c, answer3c = await agent([taskInfo, thinking2, answer2], debate_instruction_3c, r, is_sub_task=True)
            else:
                input_infos_3c = [taskInfo, thinking2, answer2] + all_thinking_3c[r-1] + all_answer_3c[r-1]
                thinking3c, answer3c = await agent(input_infos_3c, debate_instruction_3c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, eigen-decomposition of tau, thinking: {thinking3c.content}; answer: {answer3c.content}")
            all_thinking_3c[r].append(thinking3c)
            all_answer_3c[r].append(answer3c)
    final_decision_agent_3c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3c, answer3c = await final_decision_agent_3c([taskInfo, thinking2, answer2] + all_thinking_3c[-1] + all_answer_3c[-1], "Sub-task 3c: Final decision on eigen-decomposition of tau." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc_3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc_3c)
    print("Step 3c: ", sub_tasks[-1])

    debate_instruction_3d = (
        "Sub-task 3d: Compute trace Tr(tau ln tau) using eigenvalues from Sub-task 3c. "
        "Handle numerical precision and zero eigenvalues carefully. Provide detailed intermediate results. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3d = self.max_round
    all_thinking_3d = [[] for _ in range(N_max_3d)]
    all_answer_3d = [[] for _ in range(N_max_3d)]
    subtask_desc_3d = {
        "subtask_id": "subtask_3d",
        "instruction": debate_instruction_3d,
        "context": ["user query", thinking3c, answer3c],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3d):
        for i, agent in enumerate(debate_agents_3d):
            if r == 0:
                thinking3d, answer3d = await agent([taskInfo, thinking3c, answer3c], debate_instruction_3d, r, is_sub_task=True)
            else:
                input_infos_3d = [taskInfo, thinking3c, answer3c] + all_thinking_3d[r-1] + all_answer_3d[r-1]
                thinking3d, answer3d = await agent(input_infos_3d, debate_instruction_3d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compute Tr(tau ln tau), thinking: {thinking3d.content}; answer: {answer3d.content}")
            all_thinking_3d[r].append(thinking3d)
            all_answer_3d[r].append(answer3d)
    final_decision_agent_3d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3d, answer3d = await final_decision_agent_3d([taskInfo, thinking3c, answer3c] + all_thinking_3d[-1] + all_answer_3d[-1], "Sub-task 3d: Final decision on Tr(tau ln tau)." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
    subtask_desc_3d['response'] = {"thinking": thinking3d, "answer": answer3d}
    logs.append(subtask_desc_3d)
    print("Step 3d: ", sub_tasks[-1])

    # Stage 3: Calculate relative entropy measure
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_4 = (
        "Sub-task 4: Calculate the relative entropy measure of non-Gaussianity delta_b = Tr(rho ln rho) - Tr(tau ln tau) "
        "using results from Sub-tasks 3b and 3d for phi = -pi/4 and alpha = 0.5. "
        "Perform final numerical evaluation and compare with given choices: 2.48, 0, 1.38, 0.25. "
        "Select the correct answer with justification based on rigorous computation rather than assumption or consensus. "
        + reflect_inst
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3b, answer3b, thinking3d, answer3d]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", thinking3b, answer3b, thinking3d, answer3d],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, calculating relative entropy measure, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining calculation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
