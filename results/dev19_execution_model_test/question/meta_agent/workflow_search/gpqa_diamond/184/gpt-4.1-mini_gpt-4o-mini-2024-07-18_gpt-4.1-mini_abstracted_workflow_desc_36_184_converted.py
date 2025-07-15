async def forward_184(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_0 = "Sub-task 1: Extract and summarize all given information about the Hamiltonian, the operators involved, and the physical context, including the definitions and properties of the Pauli matrices and the unit vector n." + debate_instr
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking0 = [[] for _ in range(N_max_0)]
    all_answer0 = [[] for _ in range(N_max_0)]
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instruction_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking0[r-1] + all_answer0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instruction_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, extracting and summarizing info, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking0[r].append(thinking0)
            all_answer0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking0[-1] + all_answer0[-1], "Sub-task 1: Synthesize and finalize the summary of given information." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing summary, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1 = "Sub-task 2: Based on the output from Sub-task 1, analyze the operator sigma dot n to determine its eigenvalues and eigenvectors, using properties of Pauli matrices and the fact that n is a unit vector; clarify the dimension and normalization conventions of sigma."
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing operator eigenvalues, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_1 + possible_answers_1, "Sub-task 2: Synthesize and choose the most consistent and correct solutions for the operator eigenvalues." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 3: Apply the scalar factor epsilon to the eigenvalues of sigma dot n and assess whether factors of hbar/2 should be included based on the physical interpretation of sigma as spin operators or dimensionless matrices." + reflect_inst_2
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, applying scalar factor and normalization, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        critic_inst_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], "Please review and provide the limitations of provided solutions." + critic_inst_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining solution, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 4: Classify and finalize the eigenvalues of the Hamiltonian operator H, matching them to the given multiple-choice options and justifying the correct choice based on the previous analysis."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2, thinking0, answer0], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, finalizing eigenvalues classification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
