async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Parse the query text to extract the three Michael addition reactions (A, B, C), listing for each reaction all reactants (including structures), reagents, solvents, and conditions."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction1,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, parsing reactions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction2 = "Sub-task 2: Identify and assign IUPAC names (and structures) to any unspecified or ambiguous species (in particular species C) from the parsed reactions, ensuring all reactants are fully defined."
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, naming unspecified species, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: For each fully defined reaction from subtask 2, classify each species as a Michael donor (enolate precursor/nucleophile) or Michael acceptor (α,β-unsaturated carbonyl), explaining the basis for each classification."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, classifying reactants, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    sc_instruction4 = "Sub-task 4: Mechanistically predict the major product of each reaction using a Self-Consistency Chain-of-Thought: generate 3–5 distinct enolate formation → 1,4-addition → protonation pathways, validate each for atom/substituent balance, and vote to select the most consistent product structure."
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_products = []
    thinking_map4 = {}
    answer_map4 = {}
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":sc_instruction4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, predicting product pathway, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_products.append(answer4_i.content)
        thinking_map4[answer4_i.content] = thinking4_i
        answer_map4[answer4_i.content] = answer4_i
    product4_content = Counter(possible_products).most_common(1)[0][0]
    thinking4 = thinking_map4[product4_content]
    answer4 = answer_map4[product4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking":thinking4, "answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction5 = "Sub-task 5: Debate and compare the candidate product structures from subtask 4 by checking mass balance, substituent counts, stereochemistry, and correct IUPAC naming; then match the final products against the provided multiple-choice options (A–D) and determine the single correct answer."
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_thinking5 = [[] for _ in range(rounds)]
    all_answer5 = [[] for _ in range(rounds)]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction5,"context":["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"Debate"}
    for r in range(rounds):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(inputs, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final match, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the matching choice (A, B, C, or D).", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision5.id}, selected choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking":thinking5, "answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs