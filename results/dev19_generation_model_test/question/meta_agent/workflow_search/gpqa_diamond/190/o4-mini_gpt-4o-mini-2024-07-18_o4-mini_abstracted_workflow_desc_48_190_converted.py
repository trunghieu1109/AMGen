async def forward_190(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    sc_inst1 = "Sub-task 1: Summarize the starting material’s structure (3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one), list its functional groups, and outline the four reaction steps."
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":sc_inst1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents1:
        thinking, answer = await agent([taskInfo], sc_inst1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, summation, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decider1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst1 = "Given all the above thinking and answers, find the most consistent summary of the starting material and steps."
    thinking1, answer1 = await final_decider1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent summary. " + final_inst1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decider1.id}, summary decision, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    sc_inst2 = "Sub-task 2: For each reaction step described, define which bonds are formed or broken (O-alkylation, tosylhydrazone formation, Shapiro-type elimination, hydrogenolysis)."
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":sc_inst2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking, answer = await agent([taskInfo, thinking1, answer1], sc_inst2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, bond analysis, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decider2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst2 = "Given all the above thinking and answers, find the most consistent transformation definitions."
    thinking2, answer2 = await final_decider2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent definitions. " + final_inst2, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decider2.id}, transformation decision, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_inst3 = "Sub-task 3: Generate and sketch the candidate intermediate structures (products 1–3) by applying the defined transformations in order." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N3 = self.max_round
    all_thinking3 = [[] for _ in range(N3)]
    all_answer3 = [[] for _ in range(N3)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_inst3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Debate"}
    for r in range(N3):
        for agent in debate_agents3:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], debate_inst3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking, answer = await agent(inputs, debate_inst3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking3[r].append(thinking)
            all_answer3[r].append(answer)
    final_decider3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decider3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Synthesize and choose the best intermediate structures. " + final_inst3, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decider3.id}, intermediate decision, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_inst4 = "Sub-task 4: Apply the final hydrogenation/deprotection to intermediate 3 to propose the structure of product 4." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N4 = self.max_round
    all_thinking4 = [[] for _ in range(N4)]
    all_answer4 = [[] for _ in range(N4)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_inst4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"Debate"}
    for r in range(N4):
        for agent in debate_agents4:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_inst4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking, answer = await agent(inputs, debate_inst4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final_decider4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decider4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Synthesize and choose the best product-4 structure. " + final_inst4, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decider4.id}, hydrogenation decision, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    sc_inst5 = "Sub-task 5: Compare the proposed structure of product 4 against the four multiple-choice options and select the correct match."
    cot_agents5 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings5 = []
    possible_answers5 = []
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":sc_inst5,"context":["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents5:
        thinking, answer = await agent([taskInfo, thinking4, answer4], sc_inst5, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, choice comparison, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings5.append(thinking)
        possible_answers5.append(answer)
    final_decider5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst5 = "Given all the above thinking and answers, find the most consistent multiple-choice selection."
    thinking5, answer5 = await final_decider5([taskInfo, thinking4, answer4] + possible_thinkings5 + possible_answers5, "Sub-task 5: Synthesize and choose the correct multiple-choice answer. " + final_inst5, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decider5.id}, final choice decision, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs