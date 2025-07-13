async def forward_28(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: SC_CoT extract R, r, S
    sc1_instruction = "Sub-task 1: Extract the torus major radius R, minor radius r, and the sphere radius S from the user query."
    N1 = self.max_sc
    sc1_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": sc1_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in sc1_agents:
        thinking, answer = await agent([taskInfo], sc1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent extraction of R, r, S.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: SC_CoT compute D_outer and D_inner
    sc2_instruction = "Sub-task 2: Compute the center-to-center distances D_outer = S + r and D_inner = |S - r| using the extracted values."
    N2 = self.max_sc
    sc2_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": sc2_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for agent in sc2_agents:
        thinking, answer = await agent([taskInfo, thinking1, answer1], sc2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent distances D_outer and D_inner.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: Debate derive general formula
    debate_instruction3 = "Sub-task 3: Derive the general formula for the radius of the contact circle r_contact as a function of R, r, and a center-to-center distance D, and explain which sign corresponds to inner-tube vs outer-tube contact. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    N3 = self.max_round
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(N3)]
    all_answer3 = [[] for _ in range(N3)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction3, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(N3):
        for agent in debate_agents3:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking, answer = await agent(inputs, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking3[r].append(thinking)
            all_answer3[r].append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: SC_CoT apply formula to compute r_i and r_o
    sc4_instruction = "Sub-task 4: Apply the derived formula to compute r_i using D_inner with the '+' branch and r_o using D_outer with the '−' branch."
    N4 = self.max_sc
    sc4_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": sc4_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "SC_CoT"}
    for agent in sc4_agents:
        thinking, answer = await agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], sc4_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings4.append(thinking)
        possible_answers4.append(answer)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the most consistent values for r_i and r_o.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Sub-task 5: Reflexion verify r_i > r_o
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 5: Verify that the computed r_i and r_o satisfy r_i > r_o. " + reflect_inst
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Reflexion"}
    cot_inputs5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent5([taskInfo, thinking5, answer5], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    # Sub-task 6: SC_CoT compute Δr, reduce, and m+n
    sc6_instruction = "Sub-task 6: Compute the difference Δr = r_i - r_o, reduce it to lowest terms m/n, and compute m + n."
    N6 = self.max_sc
    sc6_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_thinkings6 = []
    possible_answers6 = []
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": sc6_instruction, "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "SC_CoT"}
    for agent in sc6_agents:
        thinking, answer = await agent([taskInfo, thinking4, answer4, thinking5, answer5], sc6_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings6.append(thinking)
        possible_answers6.append(answer)
    final_decision_agent6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent6([taskInfo, thinking4, answer4, thinking5, answer5] + possible_thinkings6 + possible_answers6, "Sub-task 6: Synthesize and choose the most consistent final value of m+n.", is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs