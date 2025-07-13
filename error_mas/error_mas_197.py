async def forward_197(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Formulate the expressions for α_n
    cot_sc_instruction1 = "Sub-task 1: Formulate the expressions for α_n = β_n · [SCN^-]^n for n = 0,1,2,3,4 (with α_0 = 1)."
    N1 = self.max_sc
    cot_sc_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_sc_agents1[i]([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents1[i].id}, formulating expressions, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent expressions for α_n", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Compute numerical values of α_0 through α_4
    cot_sc_instruction2 = "Sub-task 2: Compute the numerical values of α_0 through α_4 using [SCN^-] = 0.10 M and the given β_n (β1=9, β2=40, β3=63, β4=16)."
    N2 = self.max_sc
    cot_sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_sc_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents2[i].id}, computing values, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent numerical values for α_0 to α_4", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate total sum D using Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction3 = "Sub-task 3: Calculate the total sum D = ∑_{n=0}^4 α_n to serve as the denominator in the speciation." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N3 = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction3, "context": ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, initial sum calculation, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N3):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Please review and provide the limitations of provided solution." + critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, criticism round {i}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refinement round {i+1}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compute percentage P using Chain-of-Thought
    cot_instruction4 = "Sub-task 4: Compute the fraction f_2 = α_2 / D and convert it to the percentage P = 100·f_2."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, computing percentage, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask_3", "answer of subtask_3"], "agent_collaboration": "CoT", "response": {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Debate to choose closest match
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction5 = "Sub-task 5: Compare the calculated percentage P to the provided choices and identify the closest match." + debate_instr
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R5 = self.max_round
    all_thinking5 = [[] for _ in range(R5)]
    all_answer5 = [[] for _ in range(R5)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking of subtask_4", "answer of subtask_4"], "agent_collaboration": "Debate"}
    for r in range(R5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Compare and identify the closest match. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent5.id}, reasoning over debate outputs, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs