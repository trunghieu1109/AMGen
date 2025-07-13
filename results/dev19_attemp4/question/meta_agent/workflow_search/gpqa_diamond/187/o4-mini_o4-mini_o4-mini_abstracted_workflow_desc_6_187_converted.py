async def forward_187(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 0_1: Extract lattice parameters and Miller indices (SC_CoT)
    cot_sc_instruction = "Sub-task 0_1: Extract lattice parameters a and alpha, beta, gamma, and Miller indices (1,1,1) from the query."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc = {"subtask_id": "subtask_0_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent([taskInfo] + possible_thinkings + possible_answers, "Given all the above thinking and answers, find the most consistent and correct solutions for extracting parameters.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc)
    print("Step 1:", sub_tasks[-1])

    # Sub-task 0_2: Derive general expression for d_hkl (Debate)
    debate_instruction = "Sub-task 0_2: Derive the general expression for interplanar spacing d_hkl in a rhombohedral lattice using the metric tensor, a and alpha." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_think = [[] for _ in range(rounds)]
    all_ans = [[] for _ in range(rounds)]
    subtask_desc = {"subtask_id": "subtask_0_2", "instruction": debate_instruction, "context": ["user query", thinking0.content, answer0.content], "agent_collaboration": "Debate"}
    for r in range(rounds):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking0, answer0], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking0, answer0] + all_think[r-1] + all_ans[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_think[r].append(thinking)
            all_ans[r].append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_2, answer0_2 = await final_decision_agent2([taskInfo, thinking0, answer0] + all_think[-1] + all_ans[-1], "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    subtask_desc['response'] = {"thinking": thinking0_2, "answer": answer0_2}
    logs.append(subtask_desc)
    print("Step 2:", sub_tasks[-1])

    # Sub-task 1_1: Evaluate formula for (111) with a=10, alpha=30 (SC_CoT)
    cot_sc_instruction = "Sub-task 1_1: Evaluate the derived rhombohedral interplanar spacing formula for (111) by substituting a=10 and alpha=30 degrees."
    possible_thinkings = []
    possible_answers = []
    subtask_desc = {"subtask_id": "subtask_1_1", "instruction": cot_sc_instruction, "context": [thinking0_2.content, answer0_2.content], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo, thinking0_2, answer0_2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent3([taskInfo, thinking0_2, answer0_2] + possible_thinkings + possible_answers, "Given all the above thinking and answers, find the most consistent and correct numerical value.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc)
    print("Step 3:", sub_tasks[-1])

    # Sub-task 2_1: Compare with choices and select closest (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2_1: Compare the computed interplanar spacing with the provided choices and select the closest match." + reflect_inst
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc = {"subtask_id": "subtask_2_1", "instruction": cot_reflect_instruction, "context": [thinking1.content, answer1.content], "agent_collaboration": "Reflexion"}
    cot_inputs = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc)
    print("Step 4:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs