async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst1 = "Sub-task 1: Parse the query to list all chemical compounds involved and the two properties to be evaluated (tautomerism for A, optical isomerism for B)."
    thinking1, answer1 = await cot1([taskInfo], inst1, is_sub_task=True)
    agents.append(f"CoT agent {cot1.id}, parsing query, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    cot2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst2 = "Sub-task 2: Extract from the query the exact questions to be answered: (A) which of the first two compounds does not show tautomerism? and (B) which of the second pair shows optical isomerism?"
    thinking2, answer2 = await cot2([taskInfo, thinking1, answer1], inst2, is_sub_task=True)
    agents.append(f"CoT agent {cot2.id}, extracting questions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    N_sc = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    thinking_mapping = {}
    answer_mapping = {}
    possible_defs = []
    for agent in cot_agents3:
        inst3 = "Sub-task 3: Retrieve definitions of tautomerism and optical isomerism and the structural features that enable or prevent each phenomenon."
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], inst3, is_sub_task=True)
        agents.append(f"SC-CoT agent {agent.id}, retrieving definitions, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_defs.append(answer3_i.content)
        thinking_mapping[answer3_i.content] = thinking3_i
        answer_mapping[answer3_i.content] = answer3_i
    best_def = Counter(possible_defs).most_common(1)[0][0]
    thinking3 = thinking_mapping[best_def]
    answer3 = answer_mapping[best_def]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    cot4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst4 = "Sub-task 4: Analyze benzoquinone's structure to determine if it can undergo tautomerism based on the definition."
    thinking4, answer4 = await cot4([taskInfo, thinking3, answer3], inst4, is_sub_task=True)
    agents.append(f"CoT agent {cot4.id}, analyzing benzoquinone tautomerism, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    cot5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst5 = "Sub-task 5: Analyze cyclohexane-1,3,5-trione's structure to determine if it can undergo tautomerism based on the definition."
    thinking5, answer5 = await cot5([taskInfo, thinking3, answer3], inst5, is_sub_task=True)
    agents.append(f"CoT agent {cot5.id}, analyzing cyclohexane-1,3,5-trione tautomerism, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    debate_agents6 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_dr6 = self.max_round
    all_thinking6 = [[] for _ in range(N_dr6)]
    all_answer6 = [[] for _ in range(N_dr6)]
    inst6 = "Sub-task 6: Based on the tautomerism analyses, debate which compound does not exhibit tautomerism between benzoquinone and cyclohexane-1,3,5-trione."
    for r in range(N_dr6):
        for agent in debate_agents6:
            if r==0:
                thinking6_i, answer6_i = await agent([taskInfo, thinking4, answer4, thinking5, answer5], inst6, r, is_sub_task=True)
            else:
                thinking6_i, answer6_i = await agent([taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1], inst6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating tautomerism conclusion, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final_decision6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on which compound does not exhibit tautomerism.", is_sub_task=True)
    agents.append(f"Final Decision agent, concluding tautomerism nonexhibitor, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    cot7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst7 = "Sub-task 7: Analyze methyl 2-hydroxypropanoate's structure to determine if it can exhibit optical isomerism by having a chirality center."
    thinking7, answer7 = await cot7([taskInfo, thinking3, answer3], inst7, is_sub_task=True)
    agents.append(f"CoT agent {cot7.id}, analyzing methyl 2-hydroxypropanoate chirality, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    cot8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst8 = "Sub-task 8: Analyze dimethyl fumarate's structure to determine if it can exhibit optical isomerism by having a chirality center."
    thinking8, answer8 = await cot8([taskInfo, thinking3, answer3], inst8, is_sub_task=True)
    agents.append(f"CoT agent {cot8.id}, analyzing dimethyl fumarate chirality, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    debate_agents9 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_dr9 = self.max_round
    all_thinking9 = [[] for _ in range(N_dr9)]
    all_answer9 = [[] for _ in range(N_dr9)]
    inst9 = "Sub-task 9: Based on the chirality analyses, debate which compound shows optical isomerism between methyl 2-hydroxypropanoate and dimethyl fumarate."
    for r in range(N_dr9):
        for agent in debate_agents9:
            if r==0:
                thinking9_i, answer9_i = await agent([taskInfo, thinking7, answer7, thinking8, answer8], inst9, r, is_sub_task=True)
            else:
                thinking9_i, answer9_i = await agent([taskInfo, thinking7, answer7, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1], inst9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating optical isomerism conclusion, thinking: {thinking9_i.content}; answer: {answer9_i.content}")
            all_thinking9[r].append(thinking9_i)
            all_answer9[r].append(answer9_i)
    final_decision9 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on which compound shows optical isomerism.", is_sub_task=True)
    agents.append(f"Final Decision agent, concluding optical isomer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])
    cot10 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst10 = "Sub-task 10: Map the answers for part A and part B to the multiple-choice options and select the choice that matches both determinations."
    thinking10, answer10 = await cot10([taskInfo, answer6, answer9], inst10, is_sub_task=True)
    agents.append(f"CoT agent {cot10.id}, mapping answers to choices, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer