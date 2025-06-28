async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction1 = "Sub-task 1: Analyze (Z)-1-chloro-2-methylbut-1-ene: identify any stereogenic centers, assess internal planes of symmetry, and decide whether it can be optically active."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 2: Analyze (3aR,7aS,E)-8-(chloromethylene)hexahydro-4,7-methanoisobenzofuran-1,3-dione: locate all stereocenters, check for meso structures or symmetry, and determine optical activity."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible2 = []
    for agent in cot_agents2:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible2.append(answer2_i.content)
    final2 = Counter(possible2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: answers - {possible2}; final answer - {final2}")
    print("Step 2: ", sub_tasks[-1])

    cot_instruction3 = "Sub-task 3: Analyze (2R,3S)-2,3-dimethylsuccinic acid: identify its two stereocenters, assess whether internal mirror plane produces a meso form, and conclude on optical activity."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Analyze (2R,3R)-2,3-dimethylsuccinic acid: identify both stereocenters, check for enantiomeric or meso relationship, and decide on optical activity."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_instruction5 = "Sub-task 5: Analyze (R)-cyclohex-3-en-1-ol: find the single stereocenter at carbon-1, ensure no internal compensation, and determine its optical activity."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction6 = "Sub-task 6: Analyze (1s,3s,5s)-cyclohexane-1,3,5-triol: locate the three specified stereocenters, look for meso forms or symmetry elements, and decide if the molecule is optically active."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    thinking6, answer6 = await cot_agent6([taskInfo], cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], 
            "Critically evaluate the analysis for completeness, symmetry checks, and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, feedback: {feedback6.content}; correct: {correct6.content}")
        if correct6.content == "True":
            break
        thinking6, answer6 = await cot_agent6([taskInfo, thinking6, answer6, feedback6], cot_reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refined thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    cot_instruction7 = "Sub-task 7: Analyze 1-cyclopentyl-3-methylbutan-1-one: search for any stereogenic centers or chiral elements in the side chain, assess planarity or symmetry, and determine if it can be optically active."
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent7([taskInfo], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    cot_instruction8 = "Sub-task 8: Aggregate the determinations from Sub-tasks 1 through 7 to compute the total count of compounds that exhibit optical activity."
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent8([taskInfo] + sub_tasks, cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    cot_instruction9 = "Sub-task 9: Compare the computed total number of optically active compounds from Sub-task 8 with the provided choices (3, 4, 2, 5) and select the matching answer."
    cot_agent9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent9([taskInfo, thinking8, answer8], cot_instruction9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent9.id}, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer