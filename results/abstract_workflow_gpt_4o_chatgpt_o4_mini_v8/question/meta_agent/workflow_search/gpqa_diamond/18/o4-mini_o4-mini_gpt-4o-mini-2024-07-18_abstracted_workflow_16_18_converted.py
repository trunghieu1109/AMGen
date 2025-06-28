async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst1 = "Sub-task 1: Standardize and represent the reactants for Reaction A (methyl 2-oxocyclohexane-1-carboxylate and 2,4-dimethyl-1-(vinylsulfinyl)benzene) in a consistent molecular format."
    thinking1, answer1 = await cot_agent1([taskInfo], inst1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, standardizing Reaction A reactants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible2 = []
    thinking2map = {}
    answer2map = {}
    inst2 = "Sub-task 2: Identify the nucleophilic site (enolate from methyl 2-oxocyclohexane-1-carboxylate) and the electrophilic Michael acceptor (vinyl sulfoxide moiety) using the standardized structures from Sub-task 1."
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], inst2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, identifying nucleophile and electrophile, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible2.append(answer2_i.content)
        thinking2map[answer2_i.content] = thinking2_i
        answer2map[answer2_i.content] = answer2_i
    best2 = Counter(possible2).most_common(1)[0][0]
    thinking2, answer2 = thinking2map[best2], answer2map[best2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N3 = self.max_round
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    inst3 = "Sub-task 3: Predict the bonded skeleton for product A by determining the regiochemistry of the new C–C bond formed between the enolate α-carbon and the β-carbon of the vinyl group, including retention of the sulfinyl substituent."
    thinking3, answer3 = await cot_agent3(inputs3, inst3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, predicting skeleton for product A, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N3):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Critically evaluate the predicted skeleton for product A and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback on skeleton A, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(inputs3, inst3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining skeleton for product A, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst4 = "Sub-task 4: Standardize and represent the reactants for Reaction B (ethyl 2-ethylbutanoate and methyl 2-cyclopentylidene-2-phenylacetate) in a consistent molecular format."
    thinking4, answer4 = await cot_agent4([taskInfo], inst4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, standardizing Reaction B reactants, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible5 = []
    thinking5map = {}
    answer5map = {}
    inst5 = "Sub-task 5: Identify the nucleophile (enolate from ethyl 2-ethylbutanoate) and the electrophile (α,β-unsaturated ester methyl 2-cyclopentylidene-2-phenylacetate) using the standardized structures from Sub-task 4."
    for i in range(N5):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo, thinking4, answer4], inst5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, identifying nucleophile and electrophile for B, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible5.append(answer5_i.content)
        thinking5map[answer5_i.content] = thinking5_i
        answer5map[answer5_i.content] = answer5_i
    best5 = Counter(possible5).most_common(1)[0][0]
    thinking5, answer5 = thinking5map[best5], answer5map[best5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N6 = self.max_round
    inputs6 = [taskInfo, thinking4, answer4, thinking5, answer5]
    inst6 = "Sub-task 6: Predict the bonded skeleton for product B by determining where the enolate α-carbon adds to the β-carbon of the cyclopentylidene moiety, including overall ester connectivity and substitution pattern."
    thinking6, answer6 = await cot_agent6(inputs6, inst6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, predicting skeleton for product B, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N6):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], "Critically evaluate the predicted skeleton for product B and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, feedback on skeleton B, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        inputs6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent6(inputs6, inst6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining skeleton for product B, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst7 = "Sub-task 7: Compile the predicted structural skeletons of products A and B from Sub-tasks 3 and 6 into a unified format for comparison."
    thinking7, answer7 = await cot_agent7([taskInfo, thinking3, answer3, thinking6, answer6], inst7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, compiling predicted skeletons, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inst8 = "Sub-task 8: Parse and standardize the four given answer choices into molecular skeletons for both A and B components."
    thinking8, answer8 = await cot_agent8([taskInfo, answer7], inst8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, parsing answer choices, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    debate_agents9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N9 = self.max_round
    all_thinking9 = [[] for _ in range(N9)]
    all_answer9 = [[] for _ in range(N9)]
    inst9 = "Sub-task 9: Compare the predicted skeletons of A and B to each candidate's A and B structures, checking connectivity, substitution positions, and functional group placements."
    for r in range(N9):
        for i, agent in enumerate(debate_agents9):
            inputs9 = [taskInfo, thinking7, answer7, thinking8, answer8]
            if r > 0:
                inputs9 += all_thinking9[r-1] + all_answer9[r-1]
            thinking9_r, answer9_r = await agent(inputs9, inst9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing skeletons, thinking: {thinking9_r.content}; answer: {answer9_r.content}")
            all_thinking9[r].append(thinking9_r)
            all_answer9[r].append(answer9_r)
    sub_tasks.append(f"Sub-task 9 output: thinking - {all_thinking9[-1]}; answer - {all_answer9[-1]}")
    print("Step 9: ", sub_tasks[-1])
    final_agent10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_agent10([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 10: Select the choice whose structures for A and B exactly match the predicted skeletons, providing justification based on regiochemistry and substituent assignments.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent10.id}, selecting matching choice, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer