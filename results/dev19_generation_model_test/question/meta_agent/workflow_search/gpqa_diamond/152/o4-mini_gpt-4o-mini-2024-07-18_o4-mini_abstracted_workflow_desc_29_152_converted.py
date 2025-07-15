async def forward_152(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    sc_instruction = "Sub-task 1: Identify the nucleophile and electrophile for each reaction under the given conditions."
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking, answer = await cot_agents[i]([taskInfo], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, find the most consistent identification of nucleophiles and electrophiles for reactions A, B, and C."
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings + possible_answers, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction2 = "Sub-task 2: Generate the initial Michael addition product skeletons for A, B, and C based on the donor and acceptor pairings."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction2, "context": ["user query", "Sub-task 1 response"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Account for work-up and tautomerism to refine the initial skeletons into the major stable products for A, B, and C." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", "Sub-task 1 response", "Sub-task 2 response"], "agent_collaboration": "Reflexion"}
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instr4 = "Sub-task 4: Evaluate each of the four multiple-choice proposals against the refined products to score consistency in connectivity, functional groups, and numbering. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instr4, "context": ["user query", "Sub-task 3 response"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents4:
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr4, r, is_sub_task=True)
            else:
                inputs4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(inputs4, debate_instr4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    sub_tasks.append(f"Sub-task 4 output: thinking - {all_thinking4[-1][0].content}; answer - {all_answer4[-1][0].content}")
    subtask_desc4["response"] = {"thinking": all_thinking4[-1][0], "answer": all_answer4[-1][0]}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instr5 = "Sub-task 5: Select the correct choice (1–4) and provide a concise justification linking each product A, B, and C to the chosen answer. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instr5, "context": ["user query", "Sub-task 4 response"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, all_thinking4[-1][0], all_answer4[-1][0]], debate_instr5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, all_thinking4[-1][0], all_answer4[-1][0]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(inputs5, debate_instr5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent5([taskInfo, all_thinking4[-1][0], all_answer4[-1][0]] + all_thinking5[-1] + all_answer5[-1], final_instr5, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs