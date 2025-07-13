async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: SC_CoT
    sc_instruction_1 = "Sub-task 1: Analyze and classify the reactants (dimethyl malonate, methyl (E)-3-(p-tolyl)acrylate), reaction conditions (NaOEt, EtOH), and confirm the mechanism type (Michael addition)."
    cot_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thinkings_1 = []
    answers_1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":sc_instruction_1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_1[i]([taskInfo], sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyze reactants, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkings_1.append(thinking_i)
        answers_1.append(answer_i)
    final_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent_1([taskInfo]+thinkings_1+answers_1, "Sub-task 1: Synthesize the most consistent classification of mechanism and components for reaction A.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: SC_CoT
    sc_instruction_2 = "Sub-task 2: Assess how each base or nucleophile (NaOEt, piperidine) transforms its partner into the active enolate or nucleophilic species for reactions A and B."
    cot_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thinkings_2 = []
    answers_2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":sc_instruction_2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_2[i]([taskInfo, thinking1, answer1], sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, generate enolate species, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkings_2.append(thinking_i)
        answers_2.append(answer_i)
    final_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent_2([taskInfo, thinking1, answer1]+thinkings_2+answers_2, "Sub-task 2: Synthesize how bases generate active nucleophiles for subsequent addition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: SC_CoT
    sc_instruction_3 = "Sub-task 3: Derive the structures and correct IUPAC names of products A and B by applying conjugate addition of the enolate/nucleophile to the β-carbon, followed by protonation/tautomerization."
    cot_agents_3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thinkings_3 = []
    answers_3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":sc_instruction_3,"context":["user query","thinking2","answer2"],"agent_collaboration":"SC_CoT"}
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_agents_3[i]([taskInfo, thinking2, answer2], sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, propose product structures, thinking: {thinking_i.content}; answer: {answer_i.content}")
        thinkings_3.append(thinking_i)
        answers_3.append(answer_i)
    final_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent_3([taskInfo, thinking2, answer2]+thinkings_3+answers_3, "Sub-task 3: Synthesize the most consistent structures and names for products A and B.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Reverse-engineer intermediate C by considering the final product 2-(3-oxobutyl)cyclohexane-1,3-dione and deduce which cyclohexanedione tautomer served as the Michael donor. " + reflect_inst
    cot_agent_4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction,"context":["user query","thinking3","answer3"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, initial reverse-engineer, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refined reverse-engineer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Sub-task 5: Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_5 = "Sub-task 5: Compare the predicted structures and names for A, B, and C with the four provided choices and select the one that exactly matches. " + debate_instr
    debate_agents_5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction_5,"context":["user query","thinking4","answer4"],"agent_collaboration":"Debate"}
    for r in range(N_max_5):
        for agent in debate_agents_5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Synthesize the final choice." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs