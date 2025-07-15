async def forward_169(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 1: Extract and summarize the defining features of the problem: identify the spin state vector components, the S_y operator matrix, and clarify normalization requirements. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_0 = []
    all_answer_0 = []
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0):
        thinking0, answer0 = await agent([taskInfo], debate_instr_0, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, extracting problem features, thinking: {thinking0.content}; answer: {answer0.content}")
        all_thinking_0.append(thinking0)
        all_answer_0.append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_0 + all_answer_0, "Sub-task 1: Synthesize and choose the most consistent and correct summary of problem features." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing problem features, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1 = "Sub-task 1: Apply the S_y operator to the given spin state vector by performing the matrix multiplication S_y|psi> = (hbar/2)*sigma_y|psi>. Based on the output from Sub-task 1, consider/calculate potential cases of this matrix multiplication, with context from the problem and previous summary."
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, applying S_y operator, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent and correct result of applying S_y operator." , is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing S_y operator application, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 1: Calculate the expectation value <psi|S_y|psi> by computing the inner product of the bra <psi| with the vector S_y|psi>, including normalization if necessary." + reflect_inst_2
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", thinking0, answer0, thinking1, answer1],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, calculating expectation value, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining expectation value calculation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = "Sub-task 1: Evaluate the computed expectation value against the provided multiple-choice options and identify the correct answer. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3 = []
    all_answer_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3):
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_3, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, evaluating expectation value, thinking: {thinking3.content}; answer: {answer3.content}")
        all_thinking_3.append(thinking3)
        all_answer_3.append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_3 + all_answer_3, "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct multiple-choice answer, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
