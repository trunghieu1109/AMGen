async def forward_182(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Analyze the structural features of 2-formyl-5-vinylcyclohex-3-enecarboxylic acid, determining rings and unsaturations from C=C bonds, formyl, vinyl, and carboxylic acid groups."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 2: Analyze the reagent system red phosphorus + excess HI to classify likely transformations of each functional group under these strongly reducing conditions." + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking2 = [[] for _ in range(N_max)]
    all_answer2 = [[] for _ in range(N_max)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": debate_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        debate_instruction + final_instr2,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Derive the structure of the main hydrocarbon product after complete deoxygenation/reduction, determining which rings and double bonds remain or are saturated." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    max_r = self.max_round
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Reflexion"}
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(max_r):
        critic_inst = "Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        inputs3 += [thinking3, answer3, feedback]
        thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 4: Compute the index of hydrogen deficiency (degree of unsaturation) for the predicted hydrocarbon product structure."
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents4:
        thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Synthesize and choose the most consistent answer for IHD.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr5 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction5 = "Sub-task 5: Compare the calculated IHD with the provided answer choices (0, 1, 3, 5) and select the correct option." + debate_instr5
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_thinking5 = [[] for _ in range(R)]
    all_answer5 = [[] for _ in range(R)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Debate"}
    for r in range(R):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_instr5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5(
        [taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
        debate_instruction5 + final_instr5,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs