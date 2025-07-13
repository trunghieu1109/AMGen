async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0: SC_CoT for IR analysis
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    instruction1 = "Sub-task 1: Identify the functional groups present in Compound X based on the IR absorption bands (3400–2500 cm^-1, 1720 cm^-1, 1610 cm^-1, 1450 cm^-1)."
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo], instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, considering IR functional groups, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_instr1 = "Given all the above thinking and answers, find the most consistent and correct solutions for the functional groups."
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent answer for functional groups." + final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 1: SC_CoT for NMR analysis
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    instruction2 = "Sub-task 2: Analyze the ¹H NMR data (10.5 ppm (bs, 1H), 8.0 ppm (d, 2H), 7.2 ppm (d, 2H), 2.9 ppm (m, 1H), 1.7 ppm (m, 2H), 1.4 ppm (d, 3H), 0.9 ppm (t, 3H)) to determine the aromatic substitution pattern and the nature of the aliphatic substituent (sec-butyl vs. isobutyl)."
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": instruction2, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents2:
        thinking_i, answer_i = await agent([taskInfo, thinking1, answer1], instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing NMR, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings2.append(thinking_i)
        possible_answers2.append(answer_i)
    final_instr2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the aromatic pattern and aliphatic substituent."
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent answer for NMR analysis." + final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 2: Reflexion to combine IR and NMR
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    instruction3 = "Sub-task 3: Combine the IR and NMR findings to propose the full structure of Compound X (including position and identity of substituents)." + " " + reflect_inst
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": instruction3, "context": ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, initial combine, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, round {i}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent3(cot_inputs3, instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refinement round {i+1}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 3: CoT for reaction mechanism
    instruction4 = "Sub-task 4: Apply the red phosphorus/HI reaction mechanism to the deduced structure of Compound X to predict the structure of the final product (reductive decarboxylation outcome)."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": instruction4, "context": ["user query", "thinking of subtask_3", "answer of subtask_3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, mechanism, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 4: Debate to pick the final structure
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction5 = "Sub-task 5: Evaluate the four multiple-choice options against the predicted final product to select the correct structure." + " " + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking of subtask_4", "answer of subtask_4"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                input_infos5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_instr5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Final decision on the correct structure." + " " + final_instr5, is_sub_task=True)
    agents.append(f"Final Decision agent, calculation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs